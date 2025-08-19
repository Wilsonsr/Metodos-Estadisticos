import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ------------------------------------------------------------
# Configuración
# ------------------------------------------------------------
st.set_page_config(page_title="Proyección Médicos por Densidad (por 1.000)", layout="wide")
st.title("Proyección de Matriculados, Graduados y Nuevos Médicos en Medicina (por 1.000 hab.)")

# Parámetros
st.sidebar.header("🎯 Parámetros de Proyección")
densidad_objetivo = st.sidebar.slider("👨‍⚕️ Densidad deseada (médicos por 1.000 habitantes en 2035)", 3.0, 4.0, 3.7, 0.05)
tasa_graduacion = st.sidebar.slider("🎓 Tasa de graduación (matriculados que se gradúan luego de 6 años)", 0.5, 1.0, 0.8, 0.01)
tasa_cotizacion = st.sidebar.slider("📈 Tasa de graduados que comienzan a cotizar", 0.5, 1.0, 0.9, 0.01)
base_exponencial = st.sidebar.slider("📊 Base de distribución exponencial (2032–2035)", 1.00, 1.30, 1.05, 0.01)

# ------------------------------------------------------------
# Cargar datos
# ------------------------------------------------------------
archivo_csv = "https://raw.githubusercontent.com/Wilsonsr/Metodos-Estadisticos/refs/heads/main/data_medicina_graduado.csv"
df = pd.read_csv(archivo_csv)

# Normalizar columnas esperadas
df.columns = ['Anio', 'Poblacion', 'Matriculados', 'Graduados', 'Medicos_Totales',
              'Poblacion_y', 'Densidad_Medicos', 'Variacion_Medicos', 'Variacion_Porc_Medicos']
df = df.sort_values('Anio').drop_duplicates('Anio', keep='last').reset_index(drop=True)

# Convertir a numérico (evita NaN por texto)
cols_num = ['Poblacion','Matriculados','Graduados','Medicos_Totales',
            'Poblacion_y','Densidad_Medicos','Variacion_Medicos','Variacion_Porc_Medicos']
for c in cols_num:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Recalcular densidad por 1.000 (coherencia)
df['Densidad_Medicos'] = (df['Medicos_Totales'] / df['Poblacion']) * 1000

# ------------------------------------------------------------
# Si faltan 2024/2025 se crean; si existen, NO se tocan
# ------------------------------------------------------------
if 2023 in df['Anio'].values:
    mat_2023 = df.loc[df['Anio'] == 2023, 'Matriculados'].iloc[0]
    pob_2023 = df.loc[df['Anio'] == 2023, 'Poblacion'].iloc[0]

    if 2024 not in df['Anio'].values:
        df = pd.concat([df, pd.DataFrame([{
            'Anio': 2024,
            'Matriculados': mat_2023 * 1.02,
            'Poblacion': pob_2023 * 1.01
        }])], ignore_index=True)

    if 2025 not in df['Anio'].values:
        df = pd.concat([df, pd.DataFrame([{
            'Anio': 2025,
            'Matriculados': mat_2023 * (1.02**2),
            'Poblacion': pob_2023 * (1.01**2)
        }])], ignore_index=True)

df = df.sort_values('Anio').drop_duplicates('Anio', keep='last').reset_index(drop=True)

# Extender hasta 2035
if df['Anio'].max() < 2035:
    df = pd.concat([df, pd.DataFrame({'Anio': list(range(df['Anio'].max() + 1, 2036))})], ignore_index=True)
df = df.drop_duplicates('Anio', keep='last').sort_values('Anio').reset_index(drop=True)

# Asegurar población para todos los años
df['Poblacion'] = df['Poblacion'].ffill().bfill()

# ------------------------------------------------------------
# HOLT-WINTERS SOLO PARA RELLENAR 2025 SI FALTA (para que 2031 dependa de 2025)
# ------------------------------------------------------------
limite_hist = 2024 if (df['Anio'] == 2024).any() else 2023
df_hist = df[df['Anio'] <= limite_hist].dropna(subset=['Matriculados'])

df_hw = df[['Anio']].copy()
df_hw['Matriculados_HW'] = np.nan

if len(df_hist) >= 3:
    modelo_hw = ExponentialSmoothing(df_hist['Matriculados'], trend='add', seasonal=None)
    ajuste_hw = modelo_hw.fit()
    inicio_fc, fin_fc = (limite_hist + 1), 2029
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
# Cohortes y proyecciones (2031 sale de 2025; 2024 se respeta observado)
# ------------------------------------------------------------
df['Cohorte'] = df['Matriculados'].shift(6)

# Graduados: usar observados donde existan (incluye 2024); proyectar desde 2025 donde falte
df['Graduados_Proyectados'] = df['Graduados']
mask_proj = (df['Anio'] >= 2025) & (df['Graduados_Proyectados'].isna())
df.loc[mask_proj, 'Graduados_Proyectados'] = df.loc[mask_proj, 'Cohorte'] * tasa_graduacion

# Nuevos médicos desde 2025
df['Nuevos_Medicos'] = np.where(
    df['Anio'] >= 2025,
    (df['Graduados_Proyectados'] * tasa_cotizacion).round(),
    np.nan
)

# ------------------------------------------------------------
# Meta 2035 (por 1.000) y forzado 2032–2035 (retro-calcula 2026–2029)
# ------------------------------------------------------------
poblacion_2035 = df.loc[df['Anio'] == 2035, 'Poblacion'].values[0]
medicos_necesarios = int((densidad_objetivo / 1000) * poblacion_2035)
base_2031 = 175_257
medicos_faltantes = max(0, medicos_necesarios - base_2031)

def distribuir(meta, anios, base):
    pesos = np.array([base**i for i in range(len(anios))])
    pesos = pesos / pesos.sum() if pesos.sum() > 0 else np.ones_like(pesos)/len(pesos)
    dist = {anio: int(np.floor(meta * peso)) for anio, peso in zip(anios, pesos)}
    dist[anios[-1]] += meta - sum(dist.values())
    return dist

anios_meta = [2032, 2033, 2034, 2035]
dist_nuevos = distribuir(medicos_faltantes, anios_meta, base_exponencial)

# Forzar nuevos médicos 2032–2035 y retro-calcular graduados/matriculados 2026–2029
for anio in anios_meta:
    df.loc[df['Anio'] == anio, 'Nuevos_Medicos'] = dist_nuevos[anio]
    grad = dist_nuevos[anio] / tasa_cotizacion
    df.loc[df['Anio'] == anio, 'Graduados_Proyectados'] = grad
    df.loc[df['Anio'] == anio - 6, 'Matriculados'] = grad / tasa_graduacion

# ------------------------------------------------------------
# Acumulado de médicos: base 2031 conocida
# ------------------------------------------------------------
df['Medicos_Acumulados'] = np.nan
df.loc[df['Anio'] == 2031, 'Medicos_Acumulados'] = base_2031

for year in range(2032, int(df['Anio'].max()) + 1):
    if (df['Anio'] == year - 1).any() and (df['Anio'] == year).any():
        prev = df.loc[df['Anio'] == year - 1, 'Medicos_Acumulados'].values[0]
        add = df.loc[df['Anio'] == year, 'Nuevos_Medicos'].values[0]
        if pd.notna(prev) and pd.notna(add):
            df.loc[df['Anio'] == year, 'Medicos_Acumulados'] = prev + add

# Recalcular densidad lograda por coherencia más adelante
df = df.sort_values('Anio').reset_index(drop=True)

# ------------------------------------------------------------
# Holt-Winters para mostrar línea de referencia (no inyecta salvo 2025)
# ------------------------------------------------------------
# df_hw ya tiene Matriculados_HW si hubo datos suficientes
df_comp = pd.merge(
    df[['Anio', 'Matriculados']],
    df_hw[['Anio', 'Matriculados_HW']],
    on='Anio', how='left'
)
df_comp['Diferencia'] = df_comp['Matriculados'] - df_comp['Matriculados_HW']
df_dif = df_comp[df_comp['Anio'].between(2026, 2029)]
diferencia_total = int(df_dif['Diferencia'].dropna().sum()) if not df_dif.empty else 0

# ------------------------------------------------------------
# Gráfico principal
# ------------------------------------------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Anio'], y=df['Matriculados'], mode='lines+markers', name='Matriculados', line=dict(color='red')))
fig.add_trace(go.Scatter(x=df['Anio'], y=df['Graduados_Proyectados'], mode='lines+markers', name='Graduados', line=dict(color='green')))
fig.add_trace(go.Scatter(x=df['Anio'], y=df['Nuevos_Medicos'], mode='lines+markers', name='Nuevos Médicos', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df_hw['Anio'], y=df_hw['Matriculados_HW'], mode='lines+markers', name='Matriculados HW', line=dict(color='orange', dash='dash')))

# Anotaciones Δ 2026–2029
for _, row in df_dif.dropna(subset=['Diferencia']).iterrows():
    fig.add_annotation(
        x=int(row['Anio']),
        y=float(df.loc[df['Anio'] == int(row['Anio']), 'Matriculados'].values[0]),
        text=f"Δ={int(row['Diferencia']):+}",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor="gray",
        ax=0, ay=-40, font=dict(size=12, color="crimson"),
        bgcolor="white", bordercolor="crimson", borderwidth=1
    )

y_max = float(np.nanmax(df[['Matriculados','Graduados_Proyectados','Nuevos_Medicos']].values)) if len(df) else 0.0
if not np.isfinite(y_max): y_max = 0.0

fig.update_layout(
    title="📊 Proyección de Matriculados, Graduados y Nuevos Médicos (por 1.000 hab.)",
    xaxis_title="Año",
    yaxis_title="Número de Personas",
    height=600,
    yaxis=dict(tickformat=",d"),
    shapes=[
        dict(type="line", x0=2026, x1=2026, y0=0, y1=y_max, line=dict(color="gray", dash="dot"), xref='x', yref='y'),
        dict(type="line", x0=2032, x1=2032, y0=0, y1=y_max, line=dict(color="gray", dash="dot"), xref='x', yref='y')
    ]
)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# Mensajes
# ------------------------------------------------------------
st.markdown(f"""
### 🧮 Médicos requeridos en 2035 para alcanzar {densidad_objetivo} por **1.000** hab:
### 👉 **{medicos_necesarios:,}** médicos necesarios en total
""")

mensaje_dif = (
    f"✅ Entre 2026 y 2029, se proyectaron **{diferencia_total:,}** matriculados **adicionales** respecto al modelo natural (Holt-Winters)."
    if diferencia_total > 0 else
    f"⚠️ Entre 2026 y 2029, hay **{abs(diferencia_total):,}** matriculados **menos** que los esperados según Holt-Winters."
)
st.markdown(f"### 📌 Diferencia acumulada 2026–2029:\n{mensaje_dif}")

# ------------------------------------------------------------
# Tablas
# ------------------------------------------------------------
st.subheader("📊 Comparación año a año (2026–2029)")
st.dataframe(df_dif.round(0))

st.subheader("📋 Tabla de resultados")
st.dataframe(df[['Anio', 'Matriculados', 'Graduados_Proyectados', 'Nuevos_Medicos', 'Medicos_Acumulados']].round(0))

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
    file_name='proyeccion_medicos_densidad_por_1000.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

# ------------------------------------------------------------
# Densidad lograda (por 1.000)
# ------------------------------------------------------------
df['Total_Estimado'] = np.where(df['Anio'] <= 2031, df['Medicos_Totales'], df['Medicos_Acumulados'])
df['Total_Estimado'] = df['Total_Estimado'].ffill()
df['Densidad_Lograda'] = (df['Total_Estimado'] / df['Poblacion']) * 1000

st.subheader("📈 Densidad lograda por 1.000 habitantes")
fig_den = go.Figure()
df_den = df[~df['Densidad_Lograda'].isna()]

fig_den.add_trace(go.Scatter(
    x=df_den['Anio'], y=df_den['Densidad_Lograda'],
    mode='lines+markers', name='Densidad lograda (por 1.000)'
))
fig_den.add_trace(go.Scatter(
    x=df['Anio'], y=[densidad_objetivo] * len(df),
    mode='lines', name=f'Objetivo {densidad_objetivo} por 1.000', line=dict(dash='dot')
))
if (df['Anio'] == 2035).any() and pd.notna(df.loc[df['Anio'] == 2035, 'Densidad_Lograda']).all():
    dens_2035 = float(df.loc[df['Anio'] == 2035, 'Densidad_Lograda'].values[0])
    fig_den.add_annotation(
        x=2035, y=dens_2035, text=f"2035: {dens_2035:.2f} por 1.000",
        showarrow=True, arrowhead=2, ax=0, ay=-40, bgcolor="white"
    )

fig_den.update_layout(
    title="Evolución de la densidad lograda (por 1.000 hab.)",
    xaxis_title="Año",
    yaxis_title="Densidad (por 1.000 hab.)",
    height=450
)
st.plotly_chart(fig_den, use_container_width=True)

# ------------------------------------------------------------
# Diagnóstico rápido (útil para verificar 2025 → 2031)
# ------------------------------------------------------------
with st.expander("🔎 Chequeo 2025 ↔ 2031"):
    cols = ['Anio','Matriculados','Cohorte','Graduados','Graduados_Proyectados','Nuevos_Medicos']
    st.dataframe(df.loc[df['Anio'].isin([2025,2031]), cols])
