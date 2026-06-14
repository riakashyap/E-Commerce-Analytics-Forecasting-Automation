"""
Plotly Dash dashboard — run with: python dashboard/app.py
Opens at http://localhost:8050
"""

import pandas as pd
from pathlib import Path
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px

OUTPUTS = Path(__file__).parent.parent / "outputs"
PROCESSED = Path(__file__).parent.parent / "data" / "processed"

forecast_df = pd.read_csv(OUTPUTS / "forecast.csv")
anomalies_df = pd.read_csv(OUTPUTS / "anomalies.csv")
metrics_df = pd.read_csv(OUTPUTS / "model_metrics.csv")
facts_df = pd.read_csv(PROCESSED / "fact_orders.csv")

# KPI values
total_revenue = facts_df["revenue"].sum()
total_orders = facts_df["order_id"].nunique()
avg_order_value = total_revenue / total_orders

app = dash.Dash(__name__)
app.title = "E-Commerce Analytics"

# ── Helper components ─────────────────────────────────────────────────────────

def kpi_card(label: str, value: str):
    return html.Div(
        style={
            "flex": "1", "backgroundColor": "white", "borderRadius": "8px",
            "padding": "20px", "boxShadow": "0 1px 4px rgba(0,0,0,0.1)", "textAlign": "center",
        },
        children=[
            html.P(label, style={"color": "#7f8c8d", "fontSize": "13px", "margin": "0 0 8px"}),
            html.H2(value, style={"color": "#2c3e50", "margin": "0", "fontSize": "22px"}),
        ],
    )


def card_style():
    return {
        "backgroundColor": "white", "borderRadius": "8px",
        "padding": "20px", "boxShadow": "0 1px 4px rgba(0,0,0,0.1)",
    }


def header_style():
    return {"color": "#2c3e50", "marginTop": "0", "marginBottom": "16px", "fontSize": "16px"}


# ── Chart builders ────────────────────────────────────────────────────────────

def build_forecast_chart() -> go.Figure:
    actuals = forecast_df[forecast_df["type"] == "actual"]
    forecast = forecast_df[forecast_df["type"] == "forecast"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actuals["year_month"], y=actuals["revenue"],
        mode="lines+markers", name="Actual Revenue",
        line={"color": "#2980b9", "width": 2},
    ))
    bridge_x = [actuals["year_month"].iloc[-1]] + list(forecast["year_month"])
    bridge_y = [actuals["revenue"].iloc[-1]] + list(forecast["revenue"])
    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_y,
        mode="lines+markers", name="Forecast",
        line={"color": "#e67e22", "width": 2, "dash": "dash"},
    ))
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Month", yaxis_title="Revenue (R$)",
        legend={"orientation": "h", "y": -0.2},
        margin={"t": 10, "b": 40},
        hovermode="x unified",
    )
    return fig


def build_model_chart() -> go.Figure:
    fig = px.bar(
        metrics_df.sort_values("mae"),
        x="model", y="mae",
        color="model",
        color_discrete_sequence=["#2980b9", "#27ae60", "#e67e22"],
        labels={"mae": "MAE (R$)", "model": "Model"},
        text_auto=".0f",
    )
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, margin={"t": 10, "b": 40},
    )
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#f4f6f9", "padding": "24px"},
    children=[

        # Header
        html.H1("E-Commerce Analytics & Forecasting",
                style={"textAlign": "center", "color": "#2c3e50", "marginBottom": "8px"}),
        html.P("Olist Brazilian E-Commerce Dataset · 2016–2018",
               style={"textAlign": "center", "color": "#7f8c8d", "marginBottom": "32px"}),

        # KPI Cards
        html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "32px"}, children=[
            kpi_card("Total Revenue", f"R${total_revenue:,.0f}"),
            kpi_card("Total Orders", f"{total_orders:,}"),
            kpi_card("Avg Order Value", f"R${avg_order_value:,.2f}"),
            kpi_card("Anomalies Flagged", f"{len(anomalies_df):,} ({len(anomalies_df)/len(facts_df)*100:.1f}%)"),
        ]),

        # Revenue forecast chart
        html.Div(style=card_style(), children=[
            html.H3("Monthly Revenue + 3-Month Forecast", style=header_style()),
            dcc.Graph(id="forecast-chart", figure=build_forecast_chart()),
        ]),

        # Anomalies + Model comparison row
        html.Div(style={"display": "flex", "gap": "16px", "marginTop": "16px"}, children=[

            html.Div(style={**card_style(), "flex": "1"}, children=[
                html.H3("Anomalies by Category", style=header_style()),
                html.Label("Filter by state:", style={"fontSize": "13px", "color": "#555"}),
                dcc.Dropdown(
                    id="state-filter",
                    options=[{"label": "All States", "value": "ALL"}] + [
                        {"label": s, "value": s}
                        for s in sorted(anomalies_df["customer_state"].dropna().unique())
                    ],
                    value="ALL",
                    clearable=False,
                    style={"marginBottom": "12px"},
                ),
                dcc.Graph(id="anomaly-chart"),
            ]),

            html.Div(style={**card_style(), "flex": "1"}, children=[
                html.H3("Model Comparison (MAE)", style=header_style()),
                dcc.Graph(id="model-chart", figure=build_model_chart()),
            ]),
        ]),
    ],
)

# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("anomaly-chart", "figure"),
    Input("state-filter", "value"),
)
def update_anomaly_chart(state: str):
    df = anomalies_df if state == "ALL" else anomalies_df[anomalies_df["customer_state"] == state]
    counts = (
        df.groupby("product_category_name_english")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(10)
    )
    fig = px.bar(
        counts, x="count", y="product_category_name_english",
        orientation="h",
        labels={"count": "Anomalies", "product_category_name_english": "Category"},
        color_discrete_sequence=["#c0392b"],
        text_auto=True,
    )
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis={"categoryorder": "total ascending"},
        margin={"t": 10, "b": 40},
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True)
