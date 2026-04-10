import os
import time
import requests
import polars as pl
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from atproto import Client
from dotenv import load_dotenv

load_dotenv()

# Ensure NLTK vader lexicon is available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# ==========================================
# BLUESKY CREDENTIALS LOADED FROM .env
# ==========================================
BLUESKY_HANDLE = os.environ.get('BLUESKY_HANDLE', 'dpnimo11.bsky.social')
BLUESKY_PASSWORD = os.environ.get('BLUESKY_PASSWORD', '')

def fetch_bluesky_posts(ticker, limit=100, max_pages=10):
    """
    Fetches historical posts from Bluesky containing a specific cashtag via atproto.
    """
    print(f"Fetching Bluesky data for {ticker} using atproto...")
    
    client = Client()
    try:
        client.login(BLUESKY_HANDLE, BLUESKY_PASSWORD)
    except Exception as e:
        print(f"Failed to login to Bluesky. Did you set your handle and app password? Error: {e}")
        return pl.DataFrame(schema={"Ticker": pl.Utf8, "Timestamp": pl.Datetime, "Text": pl.Utf8, "Sentiment": pl.Float64})
        
    posts_data = []
    cursor = None
    
    for page in range(max_pages):
        try:
            response = client.app.bsky.feed.search_posts(
                params={'q': ticker, 'limit': limit, 'cursor': cursor}
            )
            
            for post in response.posts:
                text = getattr(post.record, 'text', '')
                created_at = getattr(post.record, 'created_at', '')
                
                if text and created_at:
                    dt_utc = pd.to_datetime(created_at, utc=True).replace(tzinfo=None)
                    sentiment_score = sia.polarity_scores(text)['compound']
                    posts_data.append({
                        "Ticker": ticker,
                        "Timestamp": dt_utc,
                        "Text": text.replace('\n', ' ').strip(),
                        "Sentiment": sentiment_score
                    })
            
            cursor = getattr(response, 'cursor', None)
            if not cursor:
                break
                
            time.sleep(1) # Rate limit safety
            
        except Exception as e:
            print(f"Error fetching page {page} for {ticker}: {e}")
            break
            
    if not posts_data:
        return pl.DataFrame(schema={"Ticker": pl.Utf8, "Timestamp": pl.Datetime, "Text": pl.Utf8, "Sentiment": pl.Float64})
        
    return pl.DataFrame(posts_data)

def collect_financial_data(tickers, start_date, end_date):
    """
    Fetches hourly price and volume data from Yahoo Finance.
    Converts yfinance Pandas dataframe to Polars.
    """
    print(f"Fetching Yahoo Finance data for {tickers}...")
    df_list = []
    for ticker in tickers:
        try:
            yf_ticker = yf.Ticker(ticker.replace("$", ""))
            pdf = yf_ticker.history(start=start_date, end=end_date, interval="1h")
            if not pdf.empty:
                pdf = pdf.reset_index()
                pdf.rename(columns={"Datetime": "Timestamp"}, inplace=True)
                pdf["Ticker"] = ticker
                pdf = pdf[["Timestamp", "Ticker", "Open", "High", "Low", "Close", "Volume"]]
                
                df = pl.from_pandas(pdf)
                df_list.append(df)
        except Exception as e:
            print(f"Error fetching YF data for {ticker}: {e}")
    
    if df_list:
        return pl.concat(df_list)
    return pl.DataFrame()

def main():
    tickers = [
        "$AAPL", "$NVDA", "$TSLA", "$GME", "$MSFT", "$AMZN", "$META", "$GOOGL", 
        "$AMD", "$SMCI", "$PLTR", "$AMC", "$INTC", "$NFLX", "$COIN", "$MSTR", 
        "$HOOD", "$BABA", "$SPY", "$QQQ", "$DJT", "$RDDT", "$ARM"
    ]
    
    # 1. Fetch Social Data
    all_posts = []
    for t in tickers:
        df_posts = fetch_bluesky_posts(t, limit=100, max_pages=50)
        all_posts.append(df_posts)
        
    social_df = pl.concat(all_posts)
    
    if social_df.is_empty():
        print("No social data fetched. Exiting.")
        return
        
    # Native datetime objects are automatically imported as Polars datetimes.
    # We assign them the correct UTC timezone context natively, truncate, and force `us` precision.
    social_df = social_df.with_columns(
        pl.col('Timestamp').dt.replace_time_zone("UTC").dt.truncate("1h").cast(pl.Datetime("us", "UTC"))
    )
    
    # Instead of deleting rows by grouping, we use a window function to count total hourly volume
    # This keeps every individual post as its own row (allowing us to reach 50,000 rows easily!)
    social_df = social_df.with_columns(
        pl.len().over(['Ticker', 'Timestamp']).alias('Post_Count')
    )
    
    # Determine the date range needed
    min_date = social_df['Timestamp'].min()
    max_date = social_df['Timestamp'].max() + timedelta(days=1)
    
    # Yahoo Finance only serves hourly data for the last 730 days.
    # Clamp min_date to 700 days ago to stay safely within that window.
    earliest_allowed = datetime.now(tz=min_date.tzinfo) - timedelta(days=700)
    if min_date < earliest_allowed:
        print(f"Clamping start date from {min_date.date()} to {earliest_allowed.date()} (YF 730-day limit)")
        min_date = earliest_allowed
    
    # Polars date conversion to string for YF
    start_str = min_date.strftime("%Y-%m-%d") if min_date else "2024-01-01"
    end_str = max_date.strftime("%Y-%m-%d") if max_date else "2024-01-02"
    
    # 2. Fetch Financial Data
    finance_df = collect_financial_data(tickers, start_str, end_str)
    
    if finance_df.is_empty():
        print("No financial data fetched. Exiting.")
        return
        
    # Convert YF timezone to UTC natively in Polars, truncate, and lock precision matching
    finance_df = finance_df.with_columns(
        pl.col('Timestamp').dt.convert_time_zone("UTC").dt.truncate("1h").cast(pl.Datetime("us", "UTC"))
    )
    
    # 3. Merge Datasets
    print("Merging datasets using Polars...")
    # Join the financial data ONTO the social data so we keep every individual post row
    merged_df = social_df.join(finance_df, on=['Ticker', 'Timestamp'], how='inner')
    
    # Handle Null values and outliers using Polars expressions
    merged_df = merged_df.drop_nulls(subset=['Close', 'Volume', 'Sentiment'])
    merged_df = merged_df.filter(pl.col('Volume') > 0)
    
    # Save & Append logic to slowly build to 50,000 threshold over multiple cron runs
    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", "merged_financial_sentiment_data.csv")
    
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print("Found existing dataset. Appending and dropping duplicates...")
        existing_df = pl.read_csv(output_path)
        # Align datatypes on the incoming existing data
        existing_df = existing_df.with_columns(
            pl.col('Timestamp').str.replace(" UTC", "").str.replace("Z", "").str.to_datetime().dt.replace_time_zone("UTC").dt.truncate("1h").cast(pl.Datetime("us", "UTC"))
        )
        # Combine and deduplicate
        merged_df = pl.concat([existing_df, merged_df], how="vertical_relaxed").unique(subset=['Ticker', 'Timestamp'], keep='last')
        
    merged_df.write_csv(output_path)
    
    print(f"Data collection complete! Dataset now has {merged_df.height} rows globally at {output_path}")

if __name__ == "__main__":
    main()
