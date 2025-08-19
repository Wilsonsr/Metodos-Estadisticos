import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Proyección Médicos por Densidad (por 1.000)", layout="wide")
st.title("Proyección de Matriculados, Graduados y Nuevos Médicos en Medicina (por 1.000 hab.)")

# Parámetros de usuario
st.sidebar.header("🎯 Parámetros de Proyección")
densidad_objetivo = st.sidebar.slider(
    "👨‍⚕️ Densidad deseada (médicos por 1.000 habitantes en 2035)",
    3.0, 4.0, 3.7, 0.05
)
tasa_graduacion = st.sidebar.slider("🎓 Tasa de graduación (matriculados que se gradúan luego de 6 años)", 0.5, 1.0, 0.8, 0.01)
tasa_cotizacion = st.sidebar.slider("📈 Tasa de graduados que comienzan a cotizar", 0.5, 1.0, 0.9, 0.01)
base_exponencial = st.sidebar.slider("📊 Base de distribución exponencial (2032–2035)", 1.00, 1.30, 1.05, 0.01)

# Cargar datos
archivo_csv = "C:/Users/wsand/Dropbox/MINSALUD/2025/Julio/medicina_graduados_primer_curso/data_medicina_graduado.csv"
df = pd.read_csv(archivo_csv)
df.columns = ['Anio', 'Poblacion', 'Matriculados', 'Graduados', 'Medicos_Totales',
              'Poblacion_y', 'Densidad_Medicos', 'Variacion_Medicos', 'Variacion_Porc_Medicos']
df = df.sort_values('Anio').reset_index(drop=True)

# Coherencia: recalcular densidad por 1.000 si la columna se usa después
df['Densidad_Medicos'] = (df['Medicos_Totales'] / df['Poblacion']) * 1000

# Proyección 2024–2025
if 2023 in df['Anio'].values:
    mat_2023 = df.loc[df['Anio'] == 2023, 'Matriculados'].values[0]
    pob_2023 = df.loc[df['Anio'] == 2023, 'Poblacion'].values[0]
    df = pd.concat([df, pd.DataFrame({'Anio': [2024, 2025]})], ignore_index=True)
    df.loc[df['Anio'] == 2024, 'Matriculados'] = mat_2023 * 1.02
    df.loc[df['Anio'] == 2025, 'Matriculados'] = mat_2023 * (1.02**2)
    df.loc[df['Anio'] == 2024, 'Poblacion'] = pob_2023 * 1.01
    df.loc[df['Anio'] == 2025, 'Poblacion'] = pob_2023 * (1.01**2)

# Extender hasta 2035
df = pd.concat([df, pd.DataFrame({'Anio': list(range(df['Anio'].max() + 1, 2036))})], ignore_index=True)
df = df.drop_duplicates('Anio').sort_values('Anio').reset_index(drop=True)

# Asegurar datos desde 2026
if df['Anio'].min() > 2026:
    df = pd.concat([pd.DataFrame({'Anio': list(range(2026, df['Anio'].min()))}), df], ignore_index=True)
    df = df.sort_values('Anio').reset_index(drop=True)

# Cálculo de graduados y nuevos médicos
df['Cohorte'] = df['Matriculados'].shift(6)
df['Graduados_Proyectados'] = np.where(df['Anio'] >= 2024, df['Cohorte'] * tasa_graduacion, df['Graduados'])
df['Nuevos_Medicos'] = np.where(df['Anio'] >= 2025, df['Graduados_Proyectados'] * tasa_cotizacion, np.nan).round()

# Acumulado
df['Medicos_Acumulados'] = np.nan
df.loc[df['Anio'] == 2031, 'Medicos_Acumulados'] = 175257  # base conocida
df['Poblacion'] = df['Poblacion'].ffill().bfill()

# Meta (por 1.000)
poblacion_2035 = df.loc[df['Anio'] == 2035, 'Poblacion'].values[0]
medicos_necesarios = int((densidad_objetivo / 1000) * poblacion_2035)
medicos_faltantes = max(0, medicos_necesarios - 175257)

# Distribución 2032–2035
def distribuir(meta, anios, base):
    pesos = np.array([base**i for i in range(len(anios))])
    pesos = pesos / pesos.sum()
    dist = {anio: int(np.floor(meta * peso)) for anio, peso in zip(anios, pesos)}
    dist[anios[-1]] += meta - sum(dist.values())
    return dist

anios_futuros = [2032, 2033, 2034, 2035]
nuevos_medicos = distribuir(medicos_faltantes, anios_futuros, base_exponencial)

for anio in anios_futuros:
    df.loc[df['Anio'] == anio, 'Nuevos_Medicos'] = nuevos_medicos[anio]
    grad = nuevos_medicos[anio] / tasa_cotizacion
    df.loc[df['Anio'] == anio, 'Graduados_Proyectados'] = grad
    df.loc[df['Anio'] == anio - 6, 'Matriculados'] = grad / tasa_graduacion

for i, row in df.iterrows():
    if row['Anio'] >= 2032:
        df.at[i, 'Medicos_Acumulados'] = df.at[i - 1, 'Medicos_Acumulados'] + row['Nuevos_Medicos']

# Proyección Holt-Winters
df_hw = df.copy()
df_hist = df_hw[df_hw['Anio'] <= 2023].dropna(subset=['Matriculados'])
modelo_hw = ExponentialSmoothing(df_hist['Matriculados'], trend='add', seasonal=None)
ajuste_hw = modelo_hw.fit()
pred_hw = ajuste_hw.forecast(6)  # 2024–2029
df_hw['Matriculados_HW'] = np.nan
df_hw.loc[df_hw['Anio'].between(2024, 2029), 'Matriculados_HW'] = pred_hw.values

# Comparación 2026–2029
df_comp = pd.merge(df[['Anio', 'Matriculados']], df_hw[['Anio', 'Matriculados_HW']], on='Anio', how='inner')
df_comp['Diferencia'] = df_comp['Matriculados'] - df_comp['Matriculados_HW']
df_dif = df_comp[df_comp['Anio'].between(2026, 2029)]
diferencia_total = int(df_dif['Diferencia'].sum())

# Gráfico principal (conteos)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Anio'], y=df['Matriculados'], mode='lines+markers', name='Matriculados', line=dict(color='red')))
fig.add_trace(go.Scatter(x=df['Anio'], y=df['Graduados_Proyectados'], mode='lines+markers', name='Graduados', line=dict(color='green')))
fig.add_trace(go.Scatter(x=df['Anio'], y=df['Nuevos_Medicos'], mode='lines+markers', name='Nuevos Médicos', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df_hw['Anio'], y=df_hw['Matriculados_HW'], mode='lines+markers', name='Matriculados HW', line=dict(color='orange', dash='dash')))

# Anotaciones de diferencia
for _, row in df_dif.iterrows():
    fig.add_annotation(
        x=row['Anio'],
        y=row['Matriculados'],
        text=f"Δ={int(row['Diferencia']):+}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor="gray",
        ax=0,
        ay=-40,
        font=dict(size=12, color="crimson"),
        bgcolor="white",
        bordercolor="crimson",
        borderwidth=1
    )

fig.update_layout(
    title="📊 Proyección de Matriculados, Graduados y Nuevos Médicos (por 1.000 hab.)",
    xaxis_title="Año",
    yaxis_title="Número de Personas",
    height=600,
    yaxis=dict(tickformat=",d"),
    shapes=[
        dict(type="line", x0=2026, x1=2026, y0=0, y1=max(df[['Matriculados', 'Graduados_Proyectados', 'Nuevos_Medicos']].max()),
             line=dict(color="gray", dash="dot"), xref='x', yref='y'),
        dict(type="line", x0=2032, x1=2032, y0=0, y1=max(df[['Matriculados', 'Graduados_Proyectados', 'Nuevos_Medicos']].max()),
             line=dict(color="gray", dash="dot"), xref='x', yref='y')
    ]
)
st.plotly_chart(fig, use_container_width=True)

# Mensaje destacado (por 1.000)
st.markdown(f"""
### 🧮 Médicos requeridos en 2035 para alcanzar {densidad_objetivo} por **1.000** hab:
### 👉 **{medicos_necesarios:,}** médicos necesarios en total
""")

# Delta acumulado
mensaje_dif = (
    f"✅ Entre 2026 y 2029, se proyectaron **{diferencia_total:,}** matriculados **adicionales** respecto al modelo natural (Holt-Winters)."
    if diferencia_total > 0 else
    f"⚠️ Entre 2026 y 2029, hay **{abs(diferencia_total):,}** matriculados **menos** que los esperados según Holt-Winters."
)
st.markdown(f"### 📌 Diferencia acumulada 2026–2029:\n{mensaje_dif}")

# Tabla comparativa
st.subheader("📊 Comparación año a año (2026–2029)")
st.dataframe(df_dif.round(0))

# Tabla general
st.subheader("📋 Tabla de resultados")
st.dataframe(df[['Anio', 'Matriculados', 'Graduados_Proyectados', 'Nuevos_Medicos', 'Medicos_Acumulados']].round(0))

# Descarga
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
# NUEVO: Gráfico independiente de DENSIDAD lograda (por 1.000)
# ------------------------------------------------------------
# Total estimado: usar histórico donde exista y proyección acumulada desde 2032
df['Total_Estimado'] = np.where(df['Anio'] <= 2031, df['Medicos_Totales'], df['Medicos_Acumulados'])
df['Total_Estimado'] = df['Total_Estimado'].ffill()

# Densidad lograda por 1.000 hab.
df['Densidad_Lograda'] = (df['Total_Estimado'] / df['Poblacion']) * 1000

st.subheader("📈 Densidad lograda por 1.000 habitantes")
fig_den = go.Figure()

# Serie de densidad lograda
df_den = df[~df['Densidad_Lograda'].isna()]
fig_den.add_trace(go.Scatter(
    x=df_den['Anio'], y=df_den['Densidad_Lograda'],
    mode='lines+markers', name='Densidad lograda (por 1.000)'
))

# Línea objetivo horizontal
fig_den.add_trace(go.Scatter(
    x=df['Anio'], y=[densidad_objetivo] * len(df),
    mode='lines', name=f'Objetivo {densidad_objetivo} por 1.000', line=dict(dash='dot')
))

# Anotar 2035
if (df['Anio'] == 2035).any():
    dens_2035 = float(df.loc[df['Anio'] == 2035, 'Densidad_Lograda'].values[0])
    fig_den.add_annotation(
        x=2035, y=dens_2035,
        text=f"2035: {dens_2035:.2f} por 1.000",
        showarrow=True, arrowhead=2, ax=0, ay=-40,
        bgcolor="white"
    )

fig_den.update_layout(
    title="Evolución de la densidad lograda (por 1.000 hab.)",
    xaxis_title="Año",
    yaxis_title="Densidad (por 1.000 hab.)",
    height=450
)

st.plotly_chart(fig_den, use_container_width=True)

