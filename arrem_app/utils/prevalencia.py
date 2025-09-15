import pandas as pd
import streamlit as st
from scipy.stats import linregress
import plotly.graph_objects as go


def calcular_prevalencia(df, sintomas, año_col="Año"):
    """Calcula la prevalencia de síntomas por año."""
    if año_col not in df.columns:
        st.error(f"La columna '{año_col}' no existe en los datos.")
        return None

    for sintoma in sintomas:
        if sintoma not in df.columns:
            st.error(f"La columna de síntoma '{sintoma}' no existe en los datos.")
            return None

    total_personas_por_año = df[año_col].value_counts().reset_index()
    total_personas_por_año.columns = [año_col, "Total_personas_año"]

    df_long = df.melt(id_vars=[año_col], value_vars=sintomas, var_name="Síntoma", value_name="Valor")

    df_prevalencia = df_long.groupby(["Síntoma", año_col, "Valor"]).agg(
        Personas_año=(año_col, 'count')
    ).reset_index()

    df_prevalencia = df_prevalencia.merge(total_personas_por_año, on=año_col, how="left")

    df_prevalencia["Prevalencia (%)"] = (df_prevalencia["Personas_año"] / df_prevalencia["Total_personas_año"] * 100).round(2)

    df_prevalencia = df_prevalencia[df_prevalencia["Valor"] == 1].drop(columns=["Valor"])

    return df_prevalencia



import streamlit as st
import plotly.graph_objects as go
from scipy.stats import linregress

def graficar_prevalencia_interactiva(df, sintoma_seleccionado="Síntoma"):
    """Genera una gráfica interactiva con doble eje usando Plotly, incluyendo regresión lineal sobre la prevalencia."""

    if df.empty:
        st.warning(f"No hay datos para graficar el síntoma {sintoma_seleccionado}.")
        return

    fig = go.Figure()

    x = df["Año"].astype(float).values
    y = df["Prevalencia (%)"].astype(float).values
    slope, intercept, r_value, _, _ = linregress(x, y)
    linea_regresion = slope * x + intercept

    # Barras: número de casos
    fig.add_trace(go.Bar(
        x=df["Año"], 
        y=df["Personas_año"],
        name="Casos",
        marker=dict(color="deepskyblue"),
        text=df["Personas_año"],
        textposition="outside",
        yaxis="y1"
    ))

    # Línea: prevalencia
    fig.add_trace(go.Scatter(
        x=df["Año"], 
        y=df["Prevalencia (%)"],
        mode="markers+lines",
        name="Prevalencia",
        marker=dict(color="orange", size=8),
        line=dict(width=2),
        text=df["Prevalencia (%)"].astype(str) + "%",
        textposition="top center",
        yaxis="y2"
    ))

    # Línea: regresión
    fig.add_trace(go.Scatter(
        x=df["Año"],
        y=linea_regresion,
        mode='lines',
        name=f'Regresión lineal (R²={r_value**2:.4f})',
        line=dict(color="red", dash="dash"),
        yaxis="y2"
    ))


    fig.update_layout(
    template="plotly_white",
    title=dict(text=f"{sintoma_seleccionado} con Regresión Lineal", x=0.02, xanchor="left"),

    xaxis=dict(title=dict(text="Año")),

    yaxis=dict(
        title=dict(text="Número de Casos", font=dict(color="deepskyblue")),
        tickfont=dict(color="deepskyblue"),
        side="left"
    ),

    yaxis2=dict(
        title=dict(text="Prevalencia (%)", font=dict(color="red")),
        tickfont=dict(color="red"),
        overlaying="y",
        side="right",
        anchor="x"
    ),

    legend=dict(
        x=1, y=1, xanchor="right", yanchor="top",
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="black", borderwidth=1
        ) 
    )


        # ... tus add_trace() y tu fig.update_layout(...) van aquí ...

    # === ANOTACIÓN CON LA ECUACIÓN (LaTeX) ===
    m, b = slope, intercept
    r2 = r_value**2
    signo = "+" if b >= 0 else "-"
    b_abs = abs(b)

    eq_latex = rf"$\hat{{y}} \;=\; {m:.3f}\,x \; {signo} \; {b_abs:.3f}\quad (R^2 = {r2:.4f})$"

    fig.add_annotation(
        x=0.99, y=0.02, xref="paper", yref="paper",
        text=eq_latex,
        showarrow=False,
        align="right",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=12)
    )

    # Render en Streamlit
    #st.plotly_chart(fig)


    st.plotly_chart(fig)










































































































































