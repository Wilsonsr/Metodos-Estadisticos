import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ------------------------------------------------------------
# Configuración
# ------------------------------------------------------------
st.set_page_config(page_title="Proyección Enfermería por Densidad (por 1.000)", layout="wide")
st.title("Proyección de Ingresantes, Titulados y Nuevos Profesionales en Enfermería (por 1.000 hab.)")

# Parámetros (alineado al flujo de Medicina, pero con L=4 y ancla 2029)
st.sidebar.header("🎯 Parámetros de Proyección")
densidad_objetivo = st.sidebar.slider("👩‍⚕️ Densidad deseada (profesionales por 1.000 habitantes en 2035)", 1.0, 3.0, 2.2, 0.05)
tasa_graduacion = st.sidebar.slider("🎓 Tasa de titulación (ingresantes que se titulan luego de 4 años)", 0.5, 1.0, 0.80, 0.01)
tasa_cotizacion = st.sidebar.slider("📈 Tasa de titulados que comienzan a cotizar", 0.3, 1.0, 0.60, 0.01)
base_exponencial = st.sidebar.slider("📊 Base de distribución exponencial (2030–2035)", 1.00, 1.30, 1.05, 0.01)

st.sidebar.markdown("---")
mostrar_detalle = st.sidebar.checkbox("Mostrar detalles técnicos (HW y sombreados)", value=False)

# ------------------------------------------------------------
# Cargar datos
# ------------------------------------------------------------
# 2024 ya está en el CSV: NO se toca.
archivo_csv = "https://raw.githubusercontent.com/Wilsonsr/Metodos-Estadisticos/refs/heads/main/data_enfermeria_graduado.csv"
df = pd.read_csv(archivo_csv)

# Normalizar columnas esperadas (acepta nombres del CSV original)
cols_csv = list(df.columns)
ren = {}
if 'Medicos_Totales' in cols_csv: ren['Medicos_Totales'] = 'Profesionales_Totales'
if 'Densidad_Medicos' in cols_csv: ren['Densidad_Medicos'] = 'Densidad'
if 'Variacion_Medicos' in cols_csv: ren['Variacion_Medicos'] = 'Var_Abs'
if 'Variacion_Porc_Medicos' in cols_csv: ren['Variacion_Porc_Medicos'] = 'Var_%'
df = df.rename(columns=ren)

esperadas = ['Anio','Poblacion','Matriculados','Graduados',
             'Profesionales_Totales','Poblacion_y','Densidad','Var_Abs','Var_%']
if len(df.columns) >= 9:
    df.columns = esperadas[:len(df.columns)]

df = df.sort_values('Anio').drop_duplicates('Anio', keep='last').reset_index(drop=True)

# A números (evita NaN por texto)
for c in ['Poblacion','Matriculados','Graduados','Profesionales_Totales','Poblacion_y','Densidad','Var_Abs','Var_%']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Coherencia histórica: densidad por 1.000 (si hay totales)
if 'Profesionales_Totales' in df.columns:
    df['Densidad'] = (df['Profesionales_Totales'] / df['Poblacion']) * 1000

# Si faltan 2024/2025 se crean; si existen en archivo, NO se tocan
if 2023 in df['Anio'].values:
    mat_2023 = df.loc[df['Anio'] == 2023, 'Matriculados'].iloc[0]
    pob_2023 = df.loc[df['Anio'] == 2023, 'Poblacion'].iloc[0]
    if 2024 not in df['Anio'].values:
        df = pd.concat([df, pd.DataFrame([{'Anio': 2024, 'Matriculados': mat_2023 * 1.02, 'Poblacion': pob_2023 * 1.01}])], ignore_index=True)
    if 2025 not in df['Anio'].values:
        df = pd.concat([df, pd.DataFrame([{'Anio': 2025, 'Matriculados': mat_2023 * (1.02**2), 'Poblacion': pob_2023 * (1.01**2)}])], ignore_index=True)

df = df.sort_values('Anio').drop_duplicates('Anio', keep='last').reset_index(drop=True)

# Extender hasta 2035 y completar población
if df['Anio'].max() < 2035:
    df = pd.concat([df, pd.DataFrame({'Anio': list(range(df['Anio'].max() + 1, 2036))})], ignore_index=True)
df = df.drop_duplicates('Anio', keep='last').sort_values('Anio').reset_index(drop=True)
df['Poblacion'] = df['Poblacion'].ffill().bfill()

# ------------------------------------------------------------
# HOLT-WINTERS SOLO PARA RELLENAR 2025 SI FALTA
# (para comparar 2026–2031 con referencia "natural")
# ------------------------------------------------------------
limite_hist = 2024 if (df['Anio'] == 2024).any() else 2023
df_hist = df[df['Anio'] <= limite_hist].dropna(subset=['Matriculados'])

df_hw = df[['Anio']].copy()
df_hw['Matriculados_HW'] = np.nan

if len(df_hist) >= 3:
    modelo_hw = ExponentialSmoothing(df_hist['Matriculados'], trend='add', seasonal=None)
    ajuste_hw = modelo_hw.fit()
    # ⬇️ CAMBIO CLAVE: proyectamos hasta 2031 para poder comparar 2026–2031
    inicio_fc, fin_fc = (limite_hist + 1), 2031
    horizonte = max(0, fin_fc - limite_hist)
    if horizonte > 0:
        pred_hw = ajuste_hw.forecast(horizonte)
        years = list(range(limite_hist + 1, fin_fc + 1))
        pred_series = pd.Series(pred_hw.values, index=years)
        df_hw.loc[df_hw['Anio'].isin(years), 'Matriculados_HW'] = df_hw['Anio'].map(pred_series)

        # Inyectar SOLO 2025 si faltara Matriculados
        if 2025 in years and (df['Anio'] == 2025).any():
            mask_2025 = df['Anio'] == 2025
            if pd.isna(df.loc[mask_2025, 'Matriculados']).all():
                df.loc[mask_2025, 'Matriculados'] = float(pred_series.loc[2025])

# ------------------------------------------------------------
# Cohortes y proyecciones (L = 4)
# ------------------------------------------------------------
L = 4
ANCLA_BASE = 2029
base_2029 = 92_681  # stock externo conocido (ajusta si cambia tu serie)

df['Cohorte'] = df['Matriculados'].shift(L)

# Titulados: usar observados donde existan (incluye 2024); proyectar desde 2025 donde falte
df['Graduados_Proyectados'] = df['Graduados']
mask_proj = (df['Anio'] >= 2025) & (df['Graduados_Proyectados'].isna())
df.loc[mask_proj, 'Graduados_Proyectados'] = df.loc[mask_proj, 'Cohorte'] * tasa_graduacion

# Nuevos profesionales desde 2025
df['Nuevos_Medicos'] = np.where(
    df['Anio'] >= 2025,
    (df['Graduados_Proyectados'] * tasa_cotizacion).round(),
    np.nan
)

# ------------------------------------------------------------
# Meta 2035 (por 1.000) y forzado 2030–2035 (retro-calcula 2026–2031)
# ------------------------------------------------------------
poblacion_2035 = df.loc[df['Anio'] == 2035, 'Poblacion'].values[0]
prof_necesarios = int((densidad_objetivo / 1000) * poblacion_2035)
prof_faltantes = max(0, prof_necesarios - base_2029)

def distribuir(meta, anios, base):
    pesos = np.array([base**i for i in range(len(anios))])
    pesos = pesos / pesos.sum() if pesos.sum() > 0 else np.ones_like(pesos)/len(pesos)
    dist = {anio: int(np.floor(meta * peso)) for anio, peso in zip(anios, pesos)}
    dist[anios[-1]] += meta - sum(dist.values())
    return dist

anios_meta = [2030, 2031, 2032, 2033, 2034, 2035]
dist_nuevos = distribuir(prof_faltantes, anios_meta, base_exponencial)

# Forzar nuevos 2030–2035 y retro-calcular titulados/ingresantes 2026–2031
for anio in anios_meta:
    df.loc[df['Anio'] == anio, 'Nuevos_Medicos'] = dist_nuevos[anio]
    grad = dist_nuevos[anio] / tasa_cotizacion
    df.loc[df['Anio'] == anio, 'Graduados_Proyectados'] = grad
    df.loc[df['Anio'] == anio - L, 'Matriculados'] = grad / tasa_graduacion  # ajusta 2026–2031

# ------------------------------------------------------------
# Acumulado de profesionales: base 2029 conocida
# ------------------------------------------------------------
df['Medicos_Acumulados'] = np.nan  # (nombre mantenido para simetría con Medicina)
df.loc[df['Anio'] == ANCLA_BASE, 'Medicos_Acumulados'] = base_2029
for year in range(ANCLA_BASE + 1, int(df['Anio'].max()) + 1):
    if (df['Anio'] == year - 1).any() and (df['Anio'] == year).any():
        prev = df.loc[df['Anio'] == year - 1, 'Medicos_Acumulados'].values[0]
        add = df.loc[df['Anio'] == year, 'Nuevos_Medicos'].values[0]
        if pd.notna(prev) and pd.notna(add):
            df.loc[df['Anio'] == year, 'Medicos_Acumulados'] = prev + add

df = df.sort_values('Anio').reset_index(drop=True)

# ------------------------------------------------------------
# Narrativa y KPIs
# ------------------------------------------------------------
st.markdown(f"""
- **Ingresantes**: estudiantes que entran cada año.  
- **Titulados**: egresan {L} años después de su ingreso.  
- **Nuevos profesionales**: titulados que comienzan a **cotizar**.  
- **2029** depende de la cohorte **2025**. **2030–2035** se ajustan para alcanzar la **densidad objetivo** ({densidad_objetivo} por 1.000).
""")

c1, c2, c3 = st.columns(3)
c1.metric("Meta 2035 (profesionales)", f"{prof_necesarios:,}")
c2.metric("Base 2029", f"{base_2029:,}")
c3.metric("Faltantes 2030–2035", f"{prof_faltantes:,}")

# ------------------------------------------------------------
# Referencia HW (comparación 2026–2031)
# ------------------------------------------------------------
df_comp = pd.merge(
    df[['Anio', 'Matriculados']],
    df_hw[['Anio', 'Matriculados_HW']],
    on='Anio', how='left'
)
df_comp['Diferencia'] = df_comp['Matriculados'] - df_comp['Matriculados_HW']

# ⬇️ CAMBIO CLAVE: ahora el rango es 2026–2031
df_dif = df_comp[df_comp['Anio'].between(2026, 2031)]
diferencia_total = int(df_dif['Diferencia'].dropna().sum()) if not df_dif.empty else 0

# ------------------------------------------------------------
# Gráfico principal SEGMENTADO por periodos (como Medicina)
# ------------------------------------------------------------
fig = go.Figure()

split_matric = 2026   # Ingresantes: a partir de aquí se ajustan por la meta
split_meta   = 2030   # Titulados/Nuevos: a partir de aquí son "ajustados"

# --- Ingresantes ---
ing_nat = df[df['Anio'] <= split_matric - 1]
ing_adj = df[df['Anio'] >= split_matric]

fig.add_trace(go.Scatter(
    x=ing_nat['Anio'], y=ing_nat['Matriculados'],
    mode='lines+markers', name='Ingresantes (≤ 2025)',
    line=dict(color='#1f77b4', width=3), marker=dict(size=7, symbol='circle'),
    legendgroup='ing',
    hovertemplate="Año %{x}<br>Ingresantes: %{y:,.0f}<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=ing_adj['Anio'], y=ing_adj['Matriculados'],
    mode='lines+markers', name='Ingresantes (≥ 2026, ajustados)',
    line=dict(color='#1f77b4', width=3, dash='dot'), marker=dict(size=7, symbol='square'),
    legendgroup='ing',
    hovertemplate="Año %{x}<br>Ingresantes (ajustados): %{y:,.0f}<extra></extra>"
))

# --- Titulados ---
tit_nat = df[df['Anio'] <= split_meta - 1]
tit_adj = df[df['Anio'] >= split_meta]

fig.add_trace(go.Scatter(
    x=tit_nat['Anio'], y=tit_nat['Graduados_Proyectados'],
    mode='lines+markers', name='Titulados (≤ 2029)',
    line=dict(color='#2ca02c', width=3), marker=dict(size=7, symbol='circle'),
    legendgroup='tit',
    hovertemplate="Año %{x}<br>Titulados: %{y:,.0f}<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=tit_adj['Anio'], y=tit_adj['Graduados_Proyectados'],
    mode='lines+markers', name='Titulados (≥ 2030, ajustados)',
    line=dict(color='#2ca02c', width=3, dash='dot'), marker=dict(size=7, symbol='square'),
    legendgroup='tit',
    hovertemplate="Año %{x}<br>Titulados (ajustados): %{y:,.0f}<extra></extra>"
))

# --- Nuevos profesionales ---
np_nat = df[df['Anio'] <= split_meta - 1]
np_adj = df[df['Anio'] >= split_meta]

fig.add_trace(go.Scatter(
    x=np_nat['Anio'], y=np_nat['Nuevos_Medicos'],
    mode='lines+markers', name='Nuevos (≤ 2029)',
    line=dict(color='#ff7f0e', width=3), marker=dict(size=7, symbol='circle'),
    legendgroup='nuevo',
    hovertemplate="Año %{x}<br>Nuevos: %{y:,.0f}<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=np_adj['Anio'], y=np_adj['Nuevos_Medicos'],
    mode='lines+markers', name='Nuevos (≥ 2030, ajustados)',
    line=dict(color='#ff7f0e', width=3, dash='dot'), marker=dict(size=7, symbol='square'),
    legendgroup='nuevo',
    hovertemplate="Año %{x}<br>Nuevos (ajustados): %{y:,.0f}<extra></extra>"
))

# --- Línea HW (opcional) ---
if mostrar_detalle:
    fig.add_trace(go.Scatter(
        x=df_hw['Anio'], y=df_hw['Matriculados_HW'],
        mode='lines+markers', name='Ingresantes (HW)',
        line=dict(color='#9467bd', width=2, dash='dash'),
        marker=dict(size=6, symbol='diamond'),
        legendgroup='hw',
        hovertemplate="Año %{x}<br>HW: %{y:,.0f}<extra></extra>"
    ))
    fig.add_vrect(x0=df['Anio'].min(), x1=2024, fillcolor="lightgray", opacity=0.07, layer="below", line_width=0,
                  annotation_text="Observado", annotation_position="top left")
    # ⬇️ CAMBIO: extender HW hasta 2031
    fig.add_vrect(x0=2025, x1=2031, fillcolor="lightblue", opacity=0.07, layer="below", line_width=0,
                  annotation_text="Proyección natural (HW)", annotation_position="top left")
    fig.add_vrect(x0=2030, x1=2035, fillcolor="lightgreen", opacity=0.07, layer="below", line_width=0,
                  annotation_text="Proyección por meta", annotation_position="top left")

y_max = float(np.nanmax(df[['Matriculados','Graduados_Proyectados','Nuevos_Medicos']].values)) if len(df) else 0.0
if not np.isfinite(y_max): y_max = 0.0

fig.add_vline(x=ANCLA_BASE, line_width=1, line_dash="dot", line_color="gray")

fig.update_layout(
    title="Proyección de Ingresantes, Titulados y Nuevos Profesionales en Enfermería (por 1.000 hab.)",
    xaxis_title="Año",
    yaxis_title="Número de personas",
    height=600,
    hovermode="x unified",
    legend_title_text="Series",
    font=dict(size=14),
    yaxis=dict(tickformat=",d", rangemode="tozero"),
    margin=dict(l=40, r=20, t=80, b=40)
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# Mensajes y tablas
# ------------------------------------------------------------
st.markdown(f"""
### 🧮 Profesionales requeridos en 2035 para alcanzar {densidad_objetivo} por **1.000** hab:
### 👉 **{prof_necesarios:,}** profesionales necesarios en total
""")

# ⬇️ CAMBIO: también el rango del mensaje
if mostrar_detalle and not df_dif.empty and df_hw['Matriculados_HW'].notna().any():
    mensaje_dif = (
        f"✅ Entre 2026 y 2031, se proyectaron **{diferencia_total:,}** ingresantes **adicionales** vs. Holt-Winters."
        if diferencia_total > 0 else
        f"⚠️ Entre 2026 y 2031, hay **{abs(diferencia_total):,}** ingresantes **menos** que los esperados según Holt-Winters."
    )
    st.markdown(f"### 📌 Diferencia acumulada 2026–2031:\n{mensaje_dif}")

st.subheader("📋 Tabla de resultados (principales)")
st.dataframe(df[['Anio', 'Matriculados', 'Graduados_Proyectados', 'Nuevos_Medicos', 'Medicos_Acumulados']].round(0))

# ⬇️ CAMBIO: subheader y rango hasta 2031
if mostrar_detalle and not df_dif.empty:
    st.subheader("📊 Comparación año a año vs HW (2026–2031)")
    st.dataframe(df_dif.round(0))

# ------------------------------------------------------------
# Descarga a Excel
# ------------------------------------------------------------
def to_excel_bytes(df_in):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_in.to_excel(writer, index=False)
    return output.getvalue()

st.download_button(
    label="📥 Descargar Excel",
    data=to_excel_bytes(df),
    file_name='proyeccion_enfermeria_densidad_por_1000.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

# ------------------------------------------------------------
# Ayuda contextual
# ------------------------------------------------------------
with st.expander("ℹ️ ¿Cómo se calcula?"):
    st.markdown(f"""
- **Titulados** del año *t* ≈ **Ingresantes** de *t-4* × tasa de titulación.
- **Nuevos profesionales** del año *t* = Titulados × tasa de cotización.
- **2029** usa la cohorte **2025**; **2030–2035** se ajustan para alcanzar la **densidad** meta ({densidad_objetivo} por 1.000).
- La línea **HW** es una referencia automática (suavizamiento); solo se usa para **rellenar 2025** si faltara.
""")

with st.expander("📝 Notas de interpretación"):
    st.info("""
- Las proyecciones cambian si se modifican las tasas o la meta.
- Una diferencia alta vs. HW en 2026–2031 no implica “mejor/peor”: refleja que **se ajustaron** las cohortes para cumplir la meta.
""")

with st.expander("🔎 Chequeo 2025 ↔ 2029"):
    cols = ['Anio','Matriculados','Cohorte','Graduados','Graduados_Proyectados','Nuevos_Medicos']
    st.dataframe(df.loc[df['Anio'].isin([2025,2029]), cols])
