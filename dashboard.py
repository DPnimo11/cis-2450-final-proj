import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import os
import base64

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Financial Sentiment Dashboard"

# Load the data
DATA_PATH = "data/processed/modeling_dataset.csv"

def load_data():
    if os.path.exists(DATA_PATH):
        df = pl.read_csv(DATA_PATH)
        # Parse timestamp string to datetime
        if df.schema.get("Timestamp") == pl.String:
            df = df.with_columns(
                pl.col("Timestamp").str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.f%z")
            )
        return df
    else:
        # Return an empty dataframe with expected columns if file is missing
        return pl.DataFrame({
            "Ticker": pl.Series(dtype=pl.String),
            "Timestamp": pl.Series(dtype=pl.Datetime),
            "Open": pl.Series(dtype=pl.Float64),
            "Close": pl.Series(dtype=pl.Float64),
            "Volume": pl.Series(dtype=pl.Int64),
            "Sentiment_Mean": pl.Series(dtype=pl.Float64),
            "Post_Count": pl.Series(dtype=pl.Int64)
        })

# Load dataset into memory
df = load_data()
tickers = df["Ticker"].unique().to_list() if not df.is_empty() else []

# Load model evaluation data
MODEL_RESULTS_PATH = "outputs/tables/model_final_results.csv"
if os.path.exists(MODEL_RESULTS_PATH):
    model_df = pl.read_csv(MODEL_RESULTS_PATH)
    # Filter for test split only
    if not model_df.is_empty() and "split" in model_df.columns:
        test_model_df = model_df.filter(pl.col("split") == "test")
    else:
        test_model_df = pl.DataFrame()
else:
    model_df = pl.DataFrame()
    test_model_df = pl.DataFrame()

# --- Layout ---

app.layout = html.Div(
    style={"fontFamily": "Inter, Roboto, sans-serif", "padding": "20px", "backgroundColor": "#f8f9fa", "minHeight": "100vh", "color": "#333"},
    children=[
        # Header
        html.Div(
            style={"textAlign": "center", "marginBottom": "30px", "padding": "20px", "backgroundColor": "#ffffff", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"},
            children=[
                html.H1("Financial Sentiment Analytics Dashboard", style={"margin": "0 0 10px 0", "color": "#2c3e50"}),
                html.P("Predicting short-term stock price movements using Bluesky sentiment and Yahoo Finance data.", style={"color": "#7f8c8d", "margin": "0"})
            ]
        ),
        
        # Tabs
        dcc.Tabs(
            id="dashboard-tabs",
            value="tab-eda",
            children=[
                dcc.Tab(label="Exploratory Data Analysis", value="tab-eda", style={"padding": "15px", "fontWeight": "bold"}, selected_style={"padding": "15px", "fontWeight": "bold", "backgroundColor": "#3498db", "color": "white", "borderTop": "3px solid #2980b9"}),
                dcc.Tab(label="Model Evaluation", value="tab-model", style={"padding": "15px", "fontWeight": "bold"}, selected_style={"padding": "15px", "fontWeight": "bold", "backgroundColor": "#3498db", "color": "white", "borderTop": "3px solid #2980b9"}),
                dcc.Tab(label="Interactive Explorer", value="tab-explorer", style={"padding": "15px", "fontWeight": "bold"}, selected_style={"padding": "15px", "fontWeight": "bold", "backgroundColor": "#3498db", "color": "white", "borderTop": "3px solid #2980b9"}),
            ],
            style={"marginBottom": "20px"}
        ),
        
        # Content container
        html.Div(id="tab-content", style={"backgroundColor": "#ffffff", "padding": "20px", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"})
    ]
)

# --- Callbacks ---

@app.callback(
    Output("tab-content", "children"),
    Input("dashboard-tabs", "value")
)
def render_tab_content(tab):
    if tab == "tab-eda":
        return render_eda_tab()
    elif tab == "tab-model":
        return render_model_tab()
    elif tab == "tab-explorer":
        return render_explorer_tab()
    return html.Div("Tab not found.")

def b64_image(image_filename):
    if os.path.exists(image_filename):
        with open(image_filename, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        return f'data:image/png;base64,{encoded}'
    return ""

def render_eda_tab():
    if df.is_empty():
        return html.Div("No data available. Please ensure data collection is complete.")
    
    # Static EDA Images
    image_paths = [
        "outputs/figures/eda_01_ticker_coverage.png",
        "outputs/figures/eda_02_sentiment_distribution.png",
        "outputs/figures/eda_03_monthly_activity_sentiment.png",
        "outputs/figures/eda_04_target_balance.png",
        "outputs/figures/eda_05_sentiment_vs_target.png",
        "outputs/figures/eda_06_example_ticker_timeline.png"
    ]
    
    image_divs = []
    for path in image_paths:
        src = b64_image(path)
        if src:
            image_divs.append(
                html.Div([
                    html.Img(src=src, style={"width": "100%", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"})
                ], style={"width": "48%", "display": "inline-block", "margin": "1%", "verticalAlign": "top"})
            )
    
    return html.Div([
        html.H3("Exploratory Data Analysis", style={"color": "#2c3e50"}),
        
        html.H4("Interactive Distributions", style={"color": "#34495e", "marginTop": "20px"}),
        html.Div([
            html.Label("Filter by Ticker:", style={"fontWeight": "bold", "marginRight": "10px"}),
            dcc.Dropdown(
                id="eda-ticker-dropdown",
                options=[{"label": "All Tickers", "value": "ALL"}] + [{"label": t, "value": t} for t in sorted(tickers)],
                value="ALL",
                style={"width": "300px", "display": "inline-block"}
            )
        ], style={"marginBottom": "20px", "display": "flex", "alignItems": "center"}),
        
        html.Div([
            dcc.Graph(id="eda-sentiment-chart", style={"display": "inline-block", "width": "48%"}),
            dcc.Graph(id="eda-volume-chart", style={"display": "inline-block", "width": "48%", "float": "right"})
        ]),
        
        html.Hr(style={"margin": "40px 0", "border": "0", "borderTop": "1px solid #ecf0f1"}),
        
        html.H4("Static EDA Reports", style={"color": "#34495e", "marginBottom": "20px"}),
        html.P("These charts were generated automatically by the feature engineering pipeline.", style={"color": "#7f8c8d"}),
        html.Div(image_divs, style={"marginTop": "20px", "display": "flex", "flexWrap": "wrap"})
    ])

def render_model_tab():
    if test_model_df.is_empty():
        return html.Div("Model evaluation data not available. Please ensure model training is complete.")
        
    scopes = test_model_df["scope"].unique().to_list()
    
    return html.Div([
        html.H3("Model Evaluation", style={"color": "#2c3e50"}),
        html.P("Compare performance metrics across different scopes and resampling strategies on the test set.", style={"color": "#7f8c8d"}),
        
        html.Div([
            html.Label("Select Scope:", style={"fontWeight": "bold", "marginRight": "10px"}),
            dcc.Dropdown(
                id="model-scope-dropdown",
                options=[{"label": s.capitalize(), "value": s} for s in scopes],
                value=scopes[0] if scopes else None,
                style={"width": "200px", "display": "inline-block"}
            )
        ], style={"marginBottom": "20px", "display": "flex", "alignItems": "center"}),
        
        # Metrics Chart
        dcc.Graph(id="model-metrics-chart", style={"marginBottom": "30px"}),
        
        # Data Table
        html.Div(id="model-metrics-table-container", style={"marginBottom": "40px"}),
        
        # Placeholders for future charts
        html.Div([
            html.Div(
                style={"border": "1px dashed #bdc3c7", "borderRadius": "8px", "padding": "50px", "textAlign": "center", "width": "45%", "display": "inline-block", "backgroundColor": "#f9fbfc"},
                children=[html.H4("Confusion Matrix", style={"color": "#95a5a6"}), html.P("Waiting for teammate's pre-computed CM data...")]
            ),
            html.Div(
                style={"border": "1px dashed #bdc3c7", "borderRadius": "8px", "padding": "50px", "textAlign": "center", "width": "45%", "display": "inline-block", "float": "right", "backgroundColor": "#f9fbfc"},
                children=[html.H4("ROC Curve", style={"color": "#95a5a6"}), html.P("Waiting for teammate's pre-computed ROC data...")]
            )
        ], style={"marginTop": "20px", "clear": "both"})
    ])

def render_explorer_tab():
    return html.Div([
        html.H3("Interactive Ticker Explorer", style={"color": "#2c3e50"}),
        html.Div([
            html.Label("Select Ticker:", style={"fontWeight": "bold", "marginRight": "10px"}),
            dcc.Dropdown(
                id="ticker-dropdown",
                options=[{"label": t, "value": t} for t in sorted(tickers)],
                value=tickers[0] if tickers else None,
                style={"width": "300px", "display": "inline-block"}
            )
        ], style={"marginBottom": "20px", "display": "flex", "alignItems": "center"}),
        
        dcc.Graph(id="ticker-timeline-chart")
    ])

@app.callback(
    [Output("eda-sentiment-chart", "figure"),
     Output("eda-volume-chart", "figure")],
    Input("eda-ticker-dropdown", "value")
)
def update_eda_charts(selected_ticker):
    if df.is_empty():
        return go.Figure(), go.Figure()
        
    filtered_df = df
    if selected_ticker and selected_ticker != "ALL":
        filtered_df = df.filter(pl.col("Ticker") == selected_ticker)
        
    # Sample to keep it fast
    n_sample = min(5000, len(filtered_df))
    pdf_sample = filtered_df.sample(n=n_sample, seed=42).to_pandas() if n_sample > 0 else filtered_df.to_pandas()
    
    fig_sentiment = px.histogram(
        pdf_sample, 
        x="Sentiment_Mean", 
        nbins=50, 
        title=f"Distribution of Hourly Sentiment ({selected_ticker})",
        color_discrete_sequence=["#3498db"]
    )
    fig_sentiment.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    
    fig_volume = px.scatter(
        pdf_sample, 
        x="Sentiment_Mean", 
        y="Volume", 
        color="Ticker" if selected_ticker == "ALL" else None, 
        opacity=0.6,
        log_y=True,
        title=f"Hourly Volume vs. Sentiment ({selected_ticker})"
    )
    fig_volume.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    if selected_ticker != "ALL":
        fig_volume.update_traces(marker_color="#e74c3c")
        
    return fig_sentiment, fig_volume

@app.callback(
    [Output("model-metrics-chart", "figure"),
     Output("model-metrics-table-container", "children")],
    Input("model-scope-dropdown", "value")
)
def update_model_tab(selected_scope):
    if not selected_scope or test_model_df.is_empty():
        return go.Figure(), html.Div()
        
    filtered = test_model_df.filter(pl.col("scope") == selected_scope)
    pdf = filtered.to_pandas()
    
    # Format model names for better display
    pdf["model_display"] = pdf["model"].str.replace("_", " ").str.title() + " (" + pdf["resampling"] + ")"
    
    # Sort by F1 score
    pdf = pdf.sort_values("f1", ascending=False)
    
    # Create grouped bar chart
    fig = go.Figure()
    
    metrics = ["f1", "roc_auc", "precision", "recall"]
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
    
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            x=pdf["model_display"],
            y=pdf[metric],
            name=metric.replace("_", " ").title(),
            marker_color=color
        ))
        
    fig.update_layout(
        title=f"Test Set Performance Metrics ({selected_scope.capitalize()} Scope)",
        barmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="Score", range=[0, 1], gridcolor="#ecf0f1")
    )
    
    # Create Data Table
    table_cols = [{"name": c.replace("_", " ").title(), "id": c} for c in ["model", "resampling", "precision", "recall", "f1", "roc_auc", "pr_auc"]]
    
    # Format numbers to 3 decimal places
    format_pdf = pdf.copy()
    for col in ["precision", "recall", "f1", "roc_auc", "pr_auc"]:
        format_pdf[col] = format_pdf[col].round(3)
        
    table = dash_table.DataTable(
        data=format_pdf.to_dict('records'),
        columns=table_cols,
        style_header={
            'backgroundColor': '#3498db',
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'left'
        },
        style_cell={
            'padding': '10px',
            'fontFamily': 'Inter, sans-serif',
            'textAlign': 'left',
            'border': '1px solid #ecf0f1'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f9f9f9'
            }
        ],
        sort_action="native"
    )
    
    return fig, table

@app.callback(
    Output("ticker-timeline-chart", "figure"),
    Input("ticker-dropdown", "value")
)
def update_timeline_chart(selected_ticker):
    if not selected_ticker or df.is_empty():
        return go.Figure()
        
    # Filter data for selected ticker
    ticker_df = df.filter(pl.col("Ticker") == selected_ticker).sort("Timestamp")
    pdf = ticker_df.to_pandas()
    
    # Create dual-axis chart
    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=pdf["Timestamp"], y=pdf["Close"], name="Close Price", line=dict(color="#2c3e50", width=2)),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Bar(x=pdf["Timestamp"], y=pdf["Sentiment_Mean"], name="Avg Sentiment", marker_color="#3498db", opacity=0.85),
        secondary_y=True,
    )
    
    fig.update_layout(
        title=f"Price and Sentiment Timeline for {selected_ticker}",
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(title_text="Date", showgrid=True, gridwidth=1, gridcolor="#ecf0f1")
    fig.update_yaxes(title_text="Close Price ($)", showgrid=False, secondary_y=False)
    fig.update_yaxes(title_text="Sentiment Score (-1 to 1)", showgrid=True, gridwidth=1, gridcolor="#ecf0f1", secondary_y=True)
    
    return fig

if __name__ == "__main__":
    app.run(debug=True, port=8050)
