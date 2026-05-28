import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

from .inference import predict

# =========================
# DATA
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "demo", "demo_vm_metrics.csv")

df = pd.read_csv(DATA_PATH)

vm_list = df["vm_id"].unique()

# =========================
# APP
# =========================

app = dash.Dash(__name__)

# =========================
# HELPERS
# =========================

def status(risk):
    if risk < 40:
        return "HEALTHY"
    elif risk < 75:
        return "DEGRADED"
    return "CRITICAL"


def color(status):
    return {
        "HEALTHY": "#22c55e",
        "DEGRADED": "#facc15",
        "CRITICAL": "#ef4444"
    }[status]
    
def explain(features):
    cpu, mem, disk, net = features

    reasons = []

    if cpu > 0.15:
        reasons.append("CPU spike")
    if mem > 0.15:
        reasons.append("Memory pressure")
    if disk > 0.15:
        reasons.append("Disk IO anomaly")
    if net > 0.15:
        reasons.append("Network spike")

    if not reasons:
        return "Normal behavior"

    return " + ".join(reasons)

# =========================
# LAYOUT
# =========================

app.layout = html.Div(style={
    "backgroundColor": "#0b1220",
    "color": "white",
    "fontFamily": "Arial",
    "minHeight": "100vh",
    "padding": "20px"
}, children=[

    # =========================
    # HEADER
    # =========================

    html.Div([
        html.H1("InfraOracle", style={"marginBottom": "5px"}),
        html.P("Predictive VM Health Monitoring System")
    ]),

    # =========================
    # FLEET SUMMARY (NEW)
    # =========================

    html.Div(id="fleet-summary", style={
        "display": "flex",
        "gap": "10px",
        "marginTop": "20px"
    }),

    # =========================
    # CONTROLS
    # =========================

    html.Div([
        dcc.Dropdown(
            id="vm-dropdown",
            options=[{"label": v, "value": v} for v in vm_list],
            value=vm_list[0],
            style={"color": "black"}
        )
    ], style={"marginTop": "20px"}),

    # =========================
    # KPI ROW
    # =========================

    html.Div(id="kpis", style={
        "display": "flex",
        "gap": "10px",
        "marginTop": "20px"
    }),

    # =========================
    # CHART GRID (NEW STRUCTURE)
    # =========================

    html.Div([

        dcc.Graph(id="cpu-chart"),
        dcc.Graph(id="mem-chart"),

    ], style={
        "display": "grid",
        "gridTemplateColumns": "1fr 1fr",
        "gap": "15px",
        "marginTop": "20px"
    }),

    html.Div([

        dcc.Graph(id="disk-chart"),
        dcc.Graph(id="net-chart"),

    ], style={
        "display": "grid",
        "gridTemplateColumns": "1fr 1fr",
        "gap": "15px",
        "marginTop": "20px"
    }),

    # =========================
    # ML OUTPUT
    # =========================

    html.Div(id="risk-output", style={
        "marginTop": "20px",
        "fontSize": "18px",
        "textAlign": "center"
    })

])

# =========================
# CALLBACK
# =========================

@app.callback(
    [
        Output("cpu-chart", "figure"),
        Output("mem-chart", "figure"),
        Output("disk-chart", "figure"),
        Output("net-chart", "figure"),
        Output("kpis", "children"),
        Output("risk-output", "children"),
        Output("fleet-summary", "children")
    ],
    [Input("vm-dropdown", "value")]
)
def update(vm):

    vm_df = df[df["vm_id"] == vm]

    latest = vm_df.tail(60)[[
        "cpu_pct",
        "mem_pct",
        "disk_io",
        "net_bytes"
    ]].values

    result = predict(latest)

    risk = result["failure_risk"]
    st = status(risk)

    # =========================
    # CHARTS (SEPARATED = IMPORTANT)
    # =========================

    cpu_fig = px.line(vm_df, x="timestamp", y="cpu_pct", title="CPU Utilization")
    mem_fig = px.line(vm_df, x="timestamp", y="mem_pct", title="Memory Utilization")
    disk_fig = px.line(vm_df, x="timestamp", y="disk_io", title="Disk IO")
    net_fig = px.line(vm_df, x="timestamp", y="net_bytes", title="Network Traffic")

    for fig in [cpu_fig, mem_fig, disk_fig, net_fig]:
        fig.update_layout(
            plot_bgcolor="#0b1220",
            paper_bgcolor="#0b1220",
            font_color="white"
        )

    # =========================
    # KPI CARDS
    # =========================

    kpis = [

        html.Div([
            html.H3(f"{risk}%"),
            html.P("Failure Risk")
        ], style={"backgroundColor": "#1e293b", "padding": "15px", "flex": "1"}),

        html.Div([
            html.H3(st),
            html.P("VM Status")
        ], style={"backgroundColor": color(st), "padding": "15px", "flex": "1"}),

        html.Div([
            html.H3(len(vm_df)),
            html.P("Data Points")
        ], style={"backgroundColor": "#1e293b", "padding": "15px", "flex": "1"})
    ]

    # =========================
    # FLEET SUMMARY (NEW)
    # =========================

    risky_vms = len(df["vm_id"].unique()) * 0.15  # placeholder logic

    fleet = [

        html.Div([
            html.H3(len(vm_list)),
            html.P("Total VMs")
        ], style={"backgroundColor": "#1e293b", "padding": "15px", "flex": "1"}),

        html.Div([
            html.H3("~15%"),
            html.P("At Risk (est)")
        ], style={"backgroundColor": "#facc15", "padding": "15px", "flex": "1"}),

        html.Div([
            html.H3("InfraOracle Live"),
            html.P("System Status")
        ], style={"backgroundColor": "#1e293b", "padding": "15px", "flex": "1"})
    ]

    # =========================
    # OUTPUT
    # =========================

    risk_text = f"""
    VM: {vm} | Status: {st} | Risk: {risk}%
    Explanation: {result.get("explanation", "N/A")}
    """

    return cpu_fig, mem_fig, disk_fig, net_fig, kpis, risk_text, fleet


# =========================
# RUN
# =========================

if __name__ == "__main__":
    app.run(debug=True)