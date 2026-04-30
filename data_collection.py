import os
import time
import requests
import polars as pl
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import torch
from transformers import pipeline
from atproto import Client
from dotenv import load_dotenv

load_dotenv()

print("Loading FinBERT...")

# Add these print statements
has_cuda = torch.cuda.is_available()
print(f"PyTorch detects CUDA (GPU): {has_cuda}")

device = 0 if has_cuda else -1
print(f"FinBERT is loading on: {'GPU' if device == 0 else 'CPU (WARNING: This will be slow)'}")

finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert", top_k=None, device=device)

# ==========================================
# BLUESKY CREDENTIALS LOADED FROM .env
# ==========================================
BLUESKY_HANDLE = os.environ.get('BLUESKY_HANDLE', 'dpnimo11.bsky.social')
BLUESKY_PASSWORD = os.environ.get('BLUESKY_PASSWORD', '')

def fetch_bluesky_posts(ticker, limit=100, max_pages=10):
    """
    Fetches historical posts from Bluesky containing a specific cashtag via atproto.
    """
    print(f"\nFetching Bluesky data for {ticker} using atproto...")
    
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
            
            texts = []
            valid_posts = []
            for post in response.posts:
                text = getattr(post.record, 'text', '')
                created_at = getattr(post.record, 'created_at', '')
                
                if text and created_at:
                    dt_utc = pd.to_datetime(created_at, utc=True).replace(tzinfo=None)
                    clean_text = text.replace('\n', ' ').strip()
                    valid_posts.append((dt_utc, clean_text))
                    texts.append(clean_text)
            
            if texts:
                # Added an indicator so you know it hasn't frozen
                print(f"  -> Processing page {page + 1}/{max_pages} ({len(texts)} posts)...")
                
                # Added batch_size=16 to prevent memory/CPU choking
                sentiments = finbert(texts, batch_size=16)
                
                for (dt_utc, clean_text), sent_results in zip(valid_posts, sentiments):
                    pos = next((x['score'] for x in sent_results if x['label'] == 'positive'), 0)
                    neg = next((x['score'] for x in sent_results if x['label'] == 'negative'), 0)
                    sentiment_score = pos - neg
                    
                    posts_data.append({
                        "Ticker": ticker,
                        "Timestamp": dt_utc,
                        "Text": clean_text,
                        "Sentiment": sentiment_score
                    })
            
            cursor = getattr(response, 'cursor', None)
            if not cursor:
                print(f"  -> No more pages available for {ticker}.")
                break
                
            time.sleep(1) # Rate limit safety
            
        except Exception as e:
            print(f"  -> Error fetching page {page} for {ticker}: {e}")
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
        df_posts = fetch_bluesky_posts(t, limit=100, max_pages=200)
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
    # We left join on Ticker and exact hour to keep weekend/overnight tweets
    merged_df = social_df.join(finance_df, on=['Ticker', 'Timestamp'], how='left')
    
    # Sort by Ticker and Timestamp so forward_fill correctly pulls Friday's close for Saturday's tweets
    merged_df = merged_df.sort(['Ticker', 'Timestamp'])
    
    # Forward fill then backward fill the financial columns to cover any gaps
    merged_df = merged_df.with_columns(
        pl.col(['Open', 'High', 'Low', 'Close', 'Volume']).forward_fill().backward_fill().over('Ticker')
    )
    
    # Drop any rows that still somehow lack finance data or sentiment
    merged_df = merged_df.drop_nulls(subset=['Close', 'Volume', 'Sentiment'])
    merged_df = merged_df.filter(pl.col('Volume') > 0)
    
    # Save & Append logic to slowly build to 50,000 threshold over multiple cron runs
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    output_path = os.path.join("data", "raw", "merged_financial_sentiment_data.csv")
    
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print("Found existing dataset. Appending and dropping duplicates...")
        existing_df = pl.read_csv(output_path)
        # Align datatypes on the incoming existing data
        existing_df = existing_df.with_columns(
            pl.col('Timestamp').str.replace(" UTC", "").str.replace("Z", "").str.to_datetime().dt.replace_time_zone("UTC").dt.truncate("1h").cast(pl.Datetime("us", "UTC"))
        )
        # Combine and deduplicate (Include Text to not drop duplicate timestamps!)
        merged_df = pl.concat([existing_df, merged_df], how="vertical_relaxed").unique(subset=['Ticker', 'Timestamp', 'Text'], keep='last')
        
    merged_df.write_csv(output_path)
    
    print(f"Data collection complete! Dataset now has {merged_df.height} rows globally at {output_path}")

if __name__ == "__main__":
    main()
