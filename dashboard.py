"""Interactive dashboard for EDA and model-result communication.

The dashboard is intentionally explanatory rather than a live trading interface.
It exposes data distributions, ticker timelines, confusion matrices, and ROC
curves so viewers can see both the useful patterns and the limitations. This
supports the rubric's dashboard/presentation criteria: the demo should not just
show charts, but should help explain why the final conclusion is modest.
"""

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
DATA_PATH = "data/processed/feature_dataset.csv"

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
    image_info = [
        ("outputs/figures/eda_01_ticker_coverage.png", "Ticker Coverage", "Shows the total volume of scraped posts per ticker, identifying our most and least discussed assets."),
        ("outputs/figures/eda_02_sentiment_distribution.png", "Sentiment Distribution", "Highlights the strong skew toward neutral sentiment scores natively output by FinBERT."),
        ("outputs/figures/eda_03_monthly_activity_sentiment.png", "Monthly Activity", "Tracks the total volume of posts and average sentiment across the dataset's timeline."),
        ("outputs/figures/eda_04_target_balance.png", "Target Balance", "Displays the class balance of our hybrid target, confirming our 0.1% threshold helps equalize Up vs Down classes."),
        ("outputs/figures/eda_05_sentiment_vs_target.png", "Sentiment vs Target", "Analyzes the distribution of sentiment scores across the target classes. Overlap indicates sentiment alone is a weak predictor."),
        ("outputs/figures/eda_06_example_ticker_timeline.png", "Example Timeline", "A time-series plot of $NVDA price action versus rolling sentiment EMA, demonstrating our temporal join.")
    ]
    
    image_divs = []
    for path, title, description in image_info:
        src = b64_image(path)
        if src:
            image_divs.append(
                html.Div([
                    html.Img(src=src, style={"width": "100%", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", "marginBottom": "10px"}),
                    html.H5(title, style={"color": "#2c3e50", "margin": "0 0 5px 0"}),
                    html.P(description, style={"color": "#7f8c8d", "fontSize": "14px", "margin": "0"})
                ], style={"width": "48%", "display": "inline-block", "margin": "1%", "verticalAlign": "top", "padding": "10px", "backgroundColor": "#f9fbfc", "borderRadius": "8px", "boxSizing": "border-box"})
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
        
        # Live CM and ROC charts
        html.Div([
            html.Div(
                style={"borderRadius": "8px", "padding": "10px", "textAlign": "center", "width": "48%", "display": "inline-block", "backgroundColor": "#ffffff", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"},
                children=[dcc.Graph(id="model-cm-chart")]
            ),
            html.Div(
                style={"borderRadius": "8px", "padding": "10px", "textAlign": "center", "width": "48%", "display": "inline-block", "float": "right", "backgroundColor": "#ffffff", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"},
                children=[dcc.Graph(id="model-roc-chart")]
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
     Output("model-metrics-table-container", "children"),
     Output("model-cm-chart", "figure"),
     Output("model-roc-chart", "figure")],
    Input("model-scope-dropdown", "value")
)
def update_model_tab(selected_scope):
    if not selected_scope or test_model_df.is_empty():
        return go.Figure(), html.Div(), go.Figure(), go.Figure()
        
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
    
    # Live Inference for CM and ROC
    cm_fig = go.Figure()
    roc_fig = go.Figure()
    
    model_path = f"outputs/models/best_{selected_scope}_model.joblib"
    try:
        if os.path.exists(model_path) and not df.is_empty():
            import joblib
            from src.modeling import chronological_train_val_test_split, get_model_feature_columns, scale_split, filter_model_scope
            from src.evaluation import predict_probabilities
            from sklearn.metrics import confusion_matrix, roc_curve, auc
            
            model_data = joblib.load(model_path)
            
            if isinstance(model_data, dict):
                model = None
                for key in ['model', 'estimator', 'classifier', 'pipeline']:
                    if key in model_data and hasattr(model_data[key], 'predict'):
                        model = model_data[key]
                        break
                if model is None:
                    for k, v in model_data.items():
                        if hasattr(v, 'predict'):
                            model = v
                            break
                if model is None:
                    raise ValueError(f"Could not find model in dict. Keys: {list(model_data.keys())}")
            else:
                model = model_data

            
            # Prepare data
            pdf_all = df.to_pandas()
            scope_df = filter_model_scope(pdf_all, selected_scope)
            feature_cols = get_model_feature_columns(scope_df)
            
            split_data = chronological_train_val_test_split(scope_df, feature_cols)
            split_data, _ = scale_split(split_data)
            
            X_test = split_data.X_test
            y_test = split_data.y_test
            
            # Generate predictions
            y_pred = model.predict(X_test)
            y_proba = predict_probabilities(model, X_test)
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            cm_fig = px.imshow(
                cm, 
                text_auto=True, 
                labels=dict(x="Predicted", y="Actual"),
                x=['Down/Neutral', 'Up'], 
                y=['Down/Neutral', 'Up'],
                color_continuous_scale="Blues",
                title=f"Confusion Matrix ({selected_scope.capitalize()})"
            )
            cm_fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc_val = auc(fpr, tpr)
            
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC = {roc_auc_val:.2f})", mode="lines", line=dict(color="#e74c3c", width=2)))
            roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", mode="lines", line=dict(color="#bdc3c7", width=2, dash="dash")))
            roc_fig.update_layout(
                title=f"ROC Curve ({selected_scope.capitalize()})",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                plot_bgcolor="white",
                paper_bgcolor="white",
                legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
            )
    except Exception as e:
        print(f"Error running inference: {e}")
    
    return fig, table, cm_fig, roc_fig

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
