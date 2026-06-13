"""
Dashboard de Demand Forecasting — Olist  |  Tema oscuro profesional
Lee reports/metrics/ y se auto-refresca cada 60 s con cada corrida de Airflow.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Página ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Olist · Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paleta y template Plotly oscuro ──────────────────────────────────────────
C = {
    "blue":   "#4f8ef7",
    "orange": "#ffa726",
    "teal":   "#00d4a0",
    "red":    "#f44336",
    "purple": "#ab47bc",
    "cyan":   "#26c6da",
    "yellow": "#ffee58",
    "card":   "#161b27",
    "border": "rgba(79,142,247,0.25)",
    "text":   "#e8eaf0",
    "sub":    "#9ea3b0",
    "bg":     "#0e1117",
    "grid":   "#1e2535",
}
PALETTE = [C["blue"], C["orange"], C["teal"], C["red"], C["purple"], C["cyan"]]

def a(hex_col: str, opacity: float) -> str:
    """Convierte color hex + opacidad a rgba() — Plotly 5.x no acepta hex de 8 dígitos."""
    r, g, b = int(hex_col[1:3], 16), int(hex_col[3:5], 16), int(hex_col[5:7], 16)
    return f"rgba({r},{g},{b},{opacity})"

def dark_layout(**kwargs):
    base = dict(
        paper_bgcolor=C["card"],
        plot_bgcolor=C["bg"],
        font=dict(color=C["text"], family="sans-serif", size=12),
        title_font=dict(size=15, color=C["text"]),
        xaxis=dict(gridcolor=C["grid"], linecolor=C["grid"], tickfont=dict(color=C["sub"])),
        yaxis=dict(gridcolor=C["grid"], linecolor=C["grid"], tickfont=dict(color=C["sub"])),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["text"])),
        margin=dict(t=50, b=40, l=50, r=20),
        hoverlabel=dict(bgcolor=C["card"], font_color=C["text"], bordercolor=C["border"]),
    )
    # Deep merge para evitar conflictos de keyword duplicado en update_layout
    for key in ("xaxis", "yaxis", "legend", "margin"):
        if key in kwargs:
            base[key] = {**base[key], **kwargs.pop(key)}
    base.update(kwargs)
    return base

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* ── Fondo y sidebar ── */
  section[data-testid="stSidebar"] {{
    background: {C['card']};
    border-right: 1px solid {C['border']};
  }}

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {{
    background: {C['card']};
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
  }}
  .stTabs [data-baseweb="tab"] {{
    border-radius: 8px;
    color: {C['sub']} !important;
    font-size: 0.88rem;
    font-weight: 500;
    padding: 8px 16px;
  }}
  .stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {C['blue']}22, {C['blue']}44) !important;
    color: {C['blue']} !important;
    border-bottom: 2px solid {C['blue']} !important;
  }}

  /* ── KPI cards ── */
  .kpi {{
    background: linear-gradient(135deg, {C['card']} 0%, #1a2235 100%);
    border: 1px solid {C['border']};
    border-radius: 14px;
    padding: 18px 14px 14px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    height: 115px;
    display: flex; flex-direction: column; justify-content: center;
    transition: border-color 0.2s;
  }}
  .kpi:hover {{ border-color: {C['blue']}; }}
  .kpi-val  {{ font-size: 1.75rem; font-weight: 800; color: {C['blue']}; line-height: 1.1; }}
  .kpi-lbl  {{ font-size: 0.70rem; color: {C['sub']}; margin-top: 6px;
               text-transform: uppercase; letter-spacing: 0.8px; }}
  .kpi-delta{{ font-size: 0.72rem; color: {C['sub']}; margin-top: 3px; }}
  .kpi-ok   {{ color: {C['teal']} !important; }}
  .kpi-warn {{ color: {C['red']}  !important; }}

  /* ── Insight box ── */
  .insight {{
    background: linear-gradient(135deg, #0d2137 0%, #0a1628 100%);
    border-left: 4px solid {C['blue']};
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    font-size: 0.87rem;
    color: #b0c4e8;
    margin-top: 10px;
    line-height: 1.6;
  }}

  /* ── Badges ── */
  .badge-ok   {{ background: {C['teal']}22; color: {C['teal']}; border: 1px solid {C['teal']}55;
                 padding: 5px 14px; border-radius: 20px; font-weight: 700; font-size: 0.83rem; }}
  .badge-fail {{ background: {C['red']}22;  color: {C['red']};  border: 1px solid {C['red']}55;
                 padding: 5px 14px; border-radius: 20px; font-weight: 700; font-size: 0.83rem; }}

  /* ── Sección header ── */
  .sec-title {{
    font-size: 1.15rem; font-weight: 700; color: {C['text']};
    border-bottom: 1px solid {C['border']}; padding-bottom: 6px; margin-bottom: 14px;
  }}

  /* ── Sidebar labels ── */
  .sb-label {{ font-size: 0.72rem; color: {C['sub']}; text-transform: uppercase;
               letter-spacing: 0.7px; margin-bottom: 2px; }}
  .sb-value {{ font-size: 0.95rem; color: {C['text']}; font-weight: 600; margin-bottom: 12px; }}
</style>
""", unsafe_allow_html=True)

# ── Rutas ────────────────────────────────────────────────────────────────────
REPORTS_DIR = Path("/app/reports/metrics")

# ── Loaders ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_summary():
    p = REPORTS_DIR / "pipeline_summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

@st.cache_data(ttl=60)
def csv(name):
    p = REPORTS_DIR / name
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def md_file(name):
    p = REPORTS_DIR / name
    return p.read_text(encoding="utf-8") if p.exists() else ""

# ── Helpers ──────────────────────────────────────────────────────────────────
def fmt(v, dec=2, suffix=""):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{round(float(v), dec)}{suffix}"

def kpi(label, value, delta=None, color=None):
    val_cls   = f"kpi-val {'kpi-ok' if color=='ok' else 'kpi-warn' if color=='warn' else ''}"
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="kpi">
      <div class="{val_cls}">{value}</div>
      <div class="kpi-lbl">{label}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight">💡 {text}</div>', unsafe_allow_html=True)

def sec(title):
    st.markdown(f'<div class="sec-title">{title}</div>', unsafe_allow_html=True)

def empty(msg="Sin datos — ejecuta el pipeline primero."):
    st.info(msg)

# ── Datos ─────────────────────────────────────────────────────────────────────
summary     = load_summary()
predictions = csv("08_selected_predictions.csv")
selection   = csv("07_final_selection_by_category.csv")
base_sum    = csv("09_summary_base_models.csv")
sel_sum     = csv("11_summary_selected_models.csv")
srch        = csv("04_arima_optuna_search_validation.csv")
hp          = csv("05_arima_hyperparameter_effect.csv")
naive_imp   = csv("03_naive_proxy_importance.csv")
arima_imp   = csv("06_arima_parameter_proxy_importance.csv")
report      = md_file("14_report_model_optimization_optuna_pro.md")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"## 📦 Olist Pipeline")
    st.markdown(f'<div style="color:{C["sub"]};font-size:0.8rem;margin-bottom:16px">Demand Forecasting Dashboard</div>', unsafe_allow_html=True)
    st.divider()

    if summary:
        m   = summary.get("metrics", {})
        bt  = m.get("backtest", {})
        val = m.get("validation", {})
        obj = summary.get("objetivo_cumplido", False)

        badge = "badge-ok" if obj else "badge-fail"
        texto = "✅ Objetivo cumplido" if obj else "❌ No cumplido"
        st.markdown(f'<div style="text-align:center;margin-bottom:16px"><span class="{badge}">{texto}</span></div>', unsafe_allow_html=True)

        for lbl, val_d in [
            ("Modelo",          summary.get("model", "—")),
            ("Categorías",      summary.get("n_categories", "—")),
            ("Train end",       summary.get("train_end", "—")),
            ("Val end",         summary.get("val_end", "—")),
            ("Último run",      summary.get("execution_date", "")[:16].replace("T", " ")),
            ("Trials ARIMA",    summary.get("optuna", {}).get("n_trials_arima", "—")),
            ("Trials Naive",    summary.get("optuna", {}).get("n_trials_naive", "—")),
        ]:
            st.markdown(f'<div class="sb-label">{lbl}</div><div class="sb-value">{val_d}</div>', unsafe_allow_html=True)

    st.divider()
    if st.button("🔄 Refrescar datos", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"Auto-refresh: 60 s\n{datetime.now().strftime('%H:%M:%S')}")

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="padding:8px 0 20px">
  <div style="font-size:2rem;font-weight:900;color:{C['text']};letter-spacing:-0.5px">
    📦 Olist Demand Forecasting
  </div>
  <div style="font-size:0.9rem;color:{C['sub']};margin-top:4px">
    Pipeline mensual · Airflow · ARIMA · Optuna · MLflow — los datos se actualizan automáticamente con cada corrida del DAG
  </div>
</div>
""", unsafe_allow_html=True)

if summary is None:
    st.warning("⏳ El pipeline aún no ha corrido. Ejecuta el DAG **olist_demand_pipeline** en Airflow y regresa aquí.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# KPIs
# ══════════════════════════════════════════════════════════════════════════════
m         = summary.get("metrics", {})
val_m     = m.get("validation", {})
bt_m      = m.get("backtest", {})
objetivo  = summary.get("objetivo_cumplido", False)
val_sel   = val_m.get("selected_mape") or val_m.get("arima_mape")
bt_sel    = bt_m.get("selected_mape")  or bt_m.get("arima_mape")
val_naive = val_m.get("naive_mape")
bt_naive  = bt_m.get("naive_mape")
impr      = bt_m.get("improvement_pp")

k1,k2,k3,k4,k5,k6 = st.columns(6)
with k1: kpi("Categorías procesadas",    str(summary.get("n_categories","—")))
with k2: kpi("Último run",               summary.get("execution_date","")[:10])
with k3: kpi("MAPE Validación",          fmt(val_sel, suffix="%"),
             delta=f"Naive: {fmt(val_naive, suffix='%')}")
with k4: kpi("MAPE Backtest",            fmt(bt_sel, suffix="%"),
             delta=f"Naive: {fmt(bt_naive, suffix='%')}",
             color="ok" if (bt_sel and bt_sel < 25) else "warn")
with k5: kpi("Mejora vs Naive",          fmt(impr, suffix=" pp"),
             color="ok" if (impr and impr > 0) else "warn")
with k6:
    badge = "badge-ok" if objetivo else "badge-fail"
    texto = "✅ Cumplido" if objetivo else "❌ No cumplido"
    st.markdown(f"""
    <div class="kpi">
      <div style="font-size:0.70rem;color:{C['sub']};text-transform:uppercase;letter-spacing:0.8px">Objetivo</div>
      <div style="margin:10px 0 4px"><span class="{badge}">{texto}</span></div>
      <div style="font-size:0.70rem;color:{C['sub']}">MAPE backtest &lt; 25%</div>
    </div>""", unsafe_allow_html=True)

st.write("")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
t1,t2,t3,t4,t5,t6,t7 = st.tabs([
    "📈  Predicciones",
    "🏆  Selección de Modelos",
    "⚔️  Base vs Optuna",
    "🔬  Búsqueda Optuna",
    "📊  Hiperparámetros",
    "🔍  Feature Importance",
    "📄  Reporte",
])

# ══ Tab 1 — Predicciones ════════════════════════════════════════════════════
with t1:
    sec("Serie real vs predicción por categoría")

    if predictions.empty:
        empty()
    else:
        predictions["year_month"] = pd.to_datetime(predictions["year_month"])
        cats = sorted(predictions["category"].dropna().unique())

        c1, c2, _ = st.columns([2, 2, 2])
        cat_sel = c1.selectbox("Categoría", cats)
        splits  = c2.multiselect("Split", ["validation", "backtest"], default=["validation", "backtest"])

        df_p = predictions[
            (predictions["category"] == cat_sel) &
            (predictions["dataset"].isin(splits))
        ].sort_values("year_month")

        if not df_p.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_p["year_month"], y=df_p["actual"],
                name="Demanda real", mode="lines+markers",
                line=dict(color=C["blue"], width=3),
                marker=dict(size=8, color=C["blue"], line=dict(color=C["bg"], width=2)),
                fill="tozeroy", fillcolor=a(C["blue"], 0.08),
            ))
            fig.add_trace(go.Scatter(
                x=df_p["year_month"], y=df_p["prediction"],
                name="Predicción", mode="lines+markers",
                line=dict(color=C["orange"], width=2.5, dash="dot"),
                marker=dict(size=9, color=C["orange"], symbol="diamond",
                            line=dict(color=C["bg"], width=2)),
            ))
            if "backtest" in splits:
                bt_d = df_p[df_p["dataset"] == "backtest"]["year_month"]
                if not bt_d.empty:
                    fig.add_vrect(
                        x0=bt_d.min(), x1=bt_d.max(),
                        fillcolor=a(C["orange"], 0.06), line_width=1,
                        line_color=a(C["orange"], 0.25),
                        annotation_text="▐ Backtest",
                        annotation_position="top left",
                        annotation_font_color=C["orange"],
                        annotation_font_size=12,
                    )
            fig.update_layout(**dark_layout(
                title=f"Demanda real vs predicción — <b>{cat_sel}</b>",
                xaxis_title="Mes", yaxis_title="Órdenes mensuales",
                legend=dict(orientation="h", y=1.1, x=1, xanchor="right"),
                height=430, hovermode="x unified",
            ))
            st.plotly_chart(fig, use_container_width=True)

            m1, m2, m3 = st.columns(3)
            for split_name in splits:
                sub = df_p[df_p["dataset"] == split_name]
                if sub.empty: continue
                mae  = (sub["actual"] - sub["prediction"]).abs().mean()
                rmse = ((sub["actual"] - sub["prediction"])**2).mean()**0.5
                mask = sub["actual"] != 0
                mape = ((sub[mask]["actual"] - sub[mask]["prediction"]).abs() / sub[mask]["actual"]).mean()*100 if mask.any() else float("nan")
                with m1: st.metric(f"MAE · {split_name}",  fmt(mae))
                with m2: st.metric(f"MAPE · {split_name}", fmt(mape, suffix="%"))
                with m3: st.metric(f"RMSE · {split_name}", fmt(rmse))

        insight(
            "El área sombreada en naranja marca el <b>período de backtest</b> — meses que el modelo nunca vio "
            "durante el entrenamiento. Una predicción pegada a la línea real en ese período indica que el modelo "
            "generaliza bien. La zona azul bajo la curva real muestra el volumen de demanda acumulado. "
            "Objetivo del proyecto: <b>MAPE &lt; 25% en backtest</b>."
        )

# ══ Tab 2 — Selección ════════════════════════════════════════════════════════
with t2:
    sec("¿Qué modelo ganó en cada categoría?")

    if selection.empty:
        empty()
    else:
        val_s = selection[selection["dataset"] == "validation"].copy() if "dataset" in selection.columns else selection.copy()
        c_pie, c_tbl = st.columns([1, 2])

        with c_pie:
            fam = (
                val_s.groupby("model_family")["category"].nunique()
                .reset_index()
                .rename(columns={"model_family":"Familia","category":"N"})
            )
            fig_pie = go.Figure(go.Pie(
                labels=fam["Familia"], values=fam["N"],
                hole=0.55,
                marker=dict(colors=[C["blue"], C["orange"]],
                            line=dict(color=C["bg"], width=3)),
                textinfo="percent+label",
                textfont=dict(size=14, color=C["text"]),
            ))
            fig_pie.update_layout(
                title="Distribución de modelos seleccionados",
                height=340, showlegend=False,
                annotations=[dict(text="<b>Winner</b>", x=0.5, y=0.5,
                                  font_size=14, font_color=C["sub"], showarrow=False)],
                **dark_layout(),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c_tbl:
            cols = [c for c in ["category","model_family","model","mae","mape","selection_score"] if c in val_s.columns]
            tbl  = val_s[cols].copy().rename(columns={
                "category":"Categoría","model_family":"Familia","model":"Modelo",
                "mae":"MAE","mape":"MAPE (%)","selection_score":"Score",
            })
            if "MAE" in tbl.columns: tbl = tbl.sort_values("MAE")
            st.dataframe(
                tbl.style
                   .format({c: "{:.2f}" for c in ["MAE","MAPE (%)","Score"] if c in tbl.columns}),
                use_container_width=True, height=320,
            )

        insight(
            "Optuna enfrenta al mejor ARIMA (con orden optimizado) contra el mejor Naive "
            "(con rezago optimizado) por cada categoría. El ganador se decide con "
            "<b>selection score = 0.6 × MAE_val + 0.3 × MAE_backtest</b>. "
            "Si Naive gana, la serie es tan volátil que el último valor conocido supera a cualquier modelo."
        )

# ══ Tab 3 — Base vs Optuna ═══════════════════════════════════════════════════
with t3:
    sec("¿Cuánto mejora Optuna respecto al ARIMA fijo (1,1,1)?")

    if base_sum.empty and sel_sum.empty:
        empty()
    else:
        for split_name, label in [("validation","Validación"), ("backtest","Backtest")]:
            rows = []
            for df_src, tag in [(base_sum,"Base (sin Optuna)"), (sel_sum,"Optuna seleccionado")]:
                if df_src.empty or "dataset" not in df_src.columns: continue
                sub = df_src[df_src["dataset"]==split_name].copy()
                sub["Config"] = tag
                rows.append(sub)
            if not rows: continue
            comb = pd.concat(rows)
            comb = comb[comb["mae"] < 5000] if "mae" in comb.columns else comb
            if comb.empty: continue

            fig_b = go.Figure()
            for tag, color in [("Base (sin Optuna)", a(C["blue"], 0.38)), ("Optuna seleccionado", C["blue"])]:
                sub = comb[comb["Config"]==tag].sort_values("mae")
                if sub.empty: continue
                fig_b.add_trace(go.Bar(
                    name=tag, x=sub["model"], y=sub["mae"],
                    marker_color=color,
                    marker_line=dict(color=C["border"], width=1),
                    text=sub["mae"].round(1), textposition="outside",
                    textfont=dict(color=C["sub"], size=10),
                ))
            fig_b.update_layout(**dark_layout(
                title=f"MAE promedio por modelo — <b>{label}</b>",
                barmode="group", xaxis_tickangle=-30, height=390,
                legend=dict(orientation="h", y=1.08),
                yaxis_title="MAE",
            ))
            st.plotly_chart(fig_b, use_container_width=True)

        insight(
            "Las barras <b>azul claro</b> son los modelos sin optimizar (ARIMA(1,1,1) fijo y naive_lag_1). "
            "Las barras <b>azul vivo</b> son los modelos seleccionados por Optuna tras 40 trials. "
            "Una barra Optuna más corta confirma que la búsqueda de hiperparámetros agrega valor real. "
            "<b>ARIMA(0,1,1)</b> es equivalente a suavizado exponencial simple — muy competitivo para demanda mensual."
        )

# ══ Tab 4 — Búsqueda Optuna ══════════════════════════════════════════════════
with t4:
    sec("Landscape de la búsqueda de hiperparámetros")

    if srch.empty:
        empty()
    else:
        df_o  = srch[srch["mae"] < 5000].copy() if "mae" in srch.columns else srch.copy()
        cats_o = sorted(df_o["category"].dropna().unique()) if "category" in df_o.columns else []

        c_f, _ = st.columns([2, 4])
        cat_o  = c_f.selectbox("Categoría", ["Todas"] + list(cats_o))
        df_of  = df_o if cat_o == "Todas" else df_o[df_o["category"]==cat_o]

        c_sc, c_best = st.columns(2)

        with c_sc:
            if all(c in df_of.columns for c in ["p","q","mae"]):
                fig_sc = go.Figure(go.Scatter(
                    x=df_of["p"], y=df_of["q"],
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=df_of["mae"],
                        colorscale="RdYlGn_r",
                        showscale=True,
                        colorbar=dict(title="MAE", tickfont=dict(color=C["sub"])),
                        line=dict(color=C["bg"], width=1),
                        opacity=0.85,
                    ),
                    text=[f"({int(r.p)},{int(r.d)},{int(r.q)}) MAE={r.mae:.1f}"
                          for _, r in df_of.iterrows()],
                    hovertemplate="%{text}<extra></extra>",
                ))
                fig_sc.update_layout(**dark_layout(
                    title="Espacio de búsqueda (p, q) — color = MAE",
                    xaxis=dict(title="Orden AR (p)", tickvals=[0,1,2]),
                    yaxis=dict(title="Orden MA (q)", tickvals=[0,1,2]),
                    height=400,
                ))
                st.plotly_chart(fig_sc, use_container_width=True)

        with c_best:
            if "category" in df_o.columns and "mae" in df_o.columns:
                best = (
                    df_o.loc[df_o.groupby("category")["mae"].idxmin()]
                    [[c for c in ["category","p","d","q","mae","mape"] if c in df_o.columns]]
                    .rename(columns={"category":"Categoría","mae":"MAE","mape":"MAPE (%)"})
                    .sort_values("MAE")
                )
                st.markdown(f"<div style='color:{C['sub']};font-size:0.8rem;font-weight:600;text-transform:uppercase;letter-spacing:0.7px;margin-bottom:8px'>Mejor orden ARIMA por categoría</div>", unsafe_allow_html=True)
                st.dataframe(
                    best.style
                        .format({c:"{:.2f}" for c in ["MAE","MAPE (%)"] if c in best.columns}
                               | {c:"{:.0f}" for c in ["p","d","q"] if c in best.columns}),
                    use_container_width=True, height=360,
                )

        insight(
            "Cada punto es una combinación (p,d,q) evaluada por Optuna en el set de validación. "
            "<b>Verde intenso</b> = MAE bajo (buena configuración); <b>rojo</b> = configuración explosiva. "
            "Optuna usa <b>TPE (Tree-structured Parzen Estimator)</b>: aprende qué regiones del espacio "
            "funcionan y concentra ahí los trials siguientes. Con 40 trials cubre el espacio "
            "(3×2×3=18 combinaciones posibles) de forma inteligente y repetida para estimar la varianza."
        )

# ══ Tab 5 — Hiperparámetros ══════════════════════════════════════════════════
with t5:
    sec("Efecto de cada hiperparámetro ARIMA sobre el MAE")

    if hp.empty:
        empty()
    else:
        df_hp  = hp[hp["mae"] < 5000].copy() if "mae" in hp.columns else hp.copy()
        params = sorted(df_hp["hyperparameter"].unique()) if "hyperparameter" in df_hp.columns else []

        if params:
            cols_hp = st.columns(len(params))
            for i, param in enumerate(params):
                sub = (
                    df_hp[df_hp["hyperparameter"]==param]
                    .groupby("value")["mae"].mean().reset_index()
                    .sort_values("value")
                )
                best_v = int(sub.loc[sub["mae"].idxmin(), "value"])
                fig_hp = go.Figure()
                fig_hp.add_trace(go.Scatter(
                    x=sub["value"], y=sub["mae"],
                    mode="lines+markers",
                    line=dict(color=C["blue"], width=3),
                    marker=dict(size=12, color=sub["mae"],
                                colorscale="RdYlGn_r",
                                line=dict(color=C["bg"], width=2),
                                showscale=False),
                    fill="tozeroy", fillcolor=a(C["blue"], 0.09),
                ))
                fig_hp.add_vline(
                    x=best_v, line_dash="dot", line_color=C["teal"], line_width=2,
                    annotation_text=f"óptimo={best_v}",
                    annotation_font_color=C["teal"],
                    annotation_position="top right",
                )
                fig_hp.update_layout(**dark_layout(
                    title=f"<b>{param.upper()}</b> vs MAE",
                    xaxis=dict(title=param.upper(), tickvals=sub["value"].tolist()),
                    yaxis_title="MAE promedio",
                    height=330,
                ))
                with cols_hp[i]:
                    st.plotly_chart(fig_hp, use_container_width=True)

        insight(
            "<b>p (AR)</b>: cuántos rezagos propios usa el modelo — captura persistencia de la demanda. "
            "<b>d</b>: número de diferenciaciones para estabilizar la serie; d=1 indica tendencia no estacionaria. "
            "<b>q (MA)</b>: errores pasados incluidos — absorbe shocks puntuales. "
            "La línea <b>teal</b> vertical marca el valor óptimo. "
            "Curvas con mínimo claro indican que el hiperparámetro es sensible y vale la pena optimizarlo."
        )

# ══ Tab 6 — Feature Importance ══════════════════════════════════════════════
with t6:
    sec("Interpretabilidad del modelo — Importancia proxy")
    st.markdown(
        f'<div style="color:{C["sub"]};font-size:0.85rem;margin-bottom:16px">'
        "ARIMA y Naive no generan feature importance tradicional (como SHAP en árboles). "
        "Se usa <b>importancia proxy</b>: para Naive, mejora relativa vs baseline naive_lag_1; "
        "para ARIMA, peso relativo de coeficientes del modelo ajustado.</div>",
        unsafe_allow_html=True,
    )

    c_n, c_a = st.columns(2)

    with c_n:
        st.markdown(f"<div style='color:{C['text']};font-weight:600;margin-bottom:8px'>Reglas Naive — ¿qué rezago predice mejor?</div>", unsafe_allow_html=True)
        if not naive_imp.empty and "proxy_importance_share_pct" in naive_imp.columns:
            agg_n = (
                naive_imp.groupby("model")["proxy_importance_share_pct"]
                .mean().reset_index()
                .sort_values("proxy_importance_share_pct", ascending=True)
            )
            fig_n = go.Figure(go.Bar(
                x=agg_n["proxy_importance_share_pct"],
                y=agg_n["model"],
                orientation="h",
                marker=dict(
                    color=agg_n["proxy_importance_share_pct"],
                    colorscale=[[0, a(C["blue"], 0.27)], [1, C["blue"]]],
                    line=dict(color=C["border"], width=1),
                ),
                text=agg_n["proxy_importance_share_pct"].round(1).astype(str) + "%",
                textposition="outside",
                textfont=dict(color=C["sub"]),
            ))
            fig_n.update_layout(**dark_layout(
                xaxis_title="Aporte proxy (%)",
                yaxis=dict(tickfont=dict(size=13, color=C["text"])),
                height=330,
            ))
            st.plotly_chart(fig_n, use_container_width=True)
            insight("Si <b>naive_ma_2</b> o <b>naive_wma_3</b> dominan, promediar los últimos meses "
                    "supera a usar solo el último valor — la demanda tiene suavidad temporal.")
        else:
            empty()

    with c_a:
        st.markdown(f"<div style='color:{C['text']};font-weight:600;margin-bottom:8px'>Coeficientes ARIMA — ¿qué componente aporta más?</div>", unsafe_allow_html=True)
        if not arima_imp.empty and "proxy_importance_share_pct" in arima_imp.columns:
            agg_a = (
                arima_imp.groupby(["term","component_type"])["proxy_importance_share_pct"]
                .mean().reset_index()
                .sort_values("proxy_importance_share_pct", ascending=True)
            )
            cmap = {"autoregressive": C["blue"], "moving_average": C["orange"],
                    "variance": C["teal"], "constant": C["purple"]}
            colors = agg_a["component_type"].map(cmap).fillna(C["sub"])
            fig_a = go.Figure(go.Bar(
                x=agg_a["proxy_importance_share_pct"],
                y=agg_a["term"],
                orientation="h",
                marker=dict(color=colors.tolist(), line=dict(color=C["border"], width=1)),
                text=agg_a["proxy_importance_share_pct"].round(1).astype(str) + "%",
                textposition="outside",
                textfont=dict(color=C["sub"]),
            ))
            # Leyenda manual por componente
            for comp, col in cmap.items():
                fig_a.add_trace(go.Bar(
                    x=[None], y=[None], name=comp,
                    marker_color=col, orientation="h",
                ))
            fig_a.update_layout(**dark_layout(
                xaxis_title="Aporte proxy (%)",
                yaxis=dict(tickfont=dict(size=13, color=C["text"])),
                legend=dict(orientation="h", y=-0.2, font=dict(color=C["sub"], size=11)),
                height=330, barmode="overlay",
            ))
            st.plotly_chart(fig_a, use_container_width=True)
            insight("<b>AR (azul)</b>: la demanda pasada predice la futura — mercado con inercia. "
                    "<b>MA (naranja)</b>: los errores recientes importan — sensible a shocks. "
                    "<b>σ² (teal)</b>: incertidumbre no explicada. AR dominante = tendencia histórica es clave.")
        else:
            empty()

# ══ Tab 7 — Reporte ══════════════════════════════════════════════════════════
with t7:
    sec("Reporte de optimización generado por el pipeline")
    st.caption("Se regenera automáticamente con cada corrida del DAG `olist_demand_pipeline` en Airflow.")
    st.write("")
    if report:
        st.markdown(report)
    else:
        empty("El reporte se generará al completar el pipeline.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    f'<div style="color:{C["sub"]};font-size:0.78rem;text-align:center">'
    f"📦 Olist Demand Forecasting · Diplomado3 · "
    f"Datos al {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
    f"Auto-refresh cada 60 s tras cada corrida de Airflow</div>",
    unsafe_allow_html=True,
)
