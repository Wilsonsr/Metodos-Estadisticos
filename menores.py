import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency, fisher_exact
import statsmodels.api as sm
import multiprocessing
import matplotlib.pyplot as plt

@st.cache_data
def cargar_datos(archivo):
    """Carga los datos desde un archivo Excel (optimizado con caché)."""
    if archivo is not None:
        return pd.read_excel(archivo)
    return None

def calcular_estadisticas_variable(df, variable_dependiente, variable_independiente):
    """Calcula estadísticas de asociación para una variable independiente."""
    tabla_contingencia = pd.crosstab(df[variable_dependiente], df[variable_independiente])
    n = np.sum(tabla_contingencia.values)

    if tabla_contingencia.empty:
        return None

    resultados = {
        'Variable Dependiente': variable_dependiente,
        'Variable Independiente': variable_independiente,
        'Chi2': None, 'p-valor': None, 'Phi^2': None, 'Cramer\'s V': None,
        'Coeficiente de Contingencia': None, 'Odds Ratio': "No aplicable",
        'IC 95% OR': "No aplicable", 'Razón de Prevalencia': "No aplicable",
        'IC 95% RP': "No aplicable", 'p-valor Fisher': "No aplicable"
    }

    chi2, p, dof, expected = chi2_contingency(tabla_contingencia)
    phi2 = chi2 / n
    r, k = tabla_contingencia.shape
    cramer_v = np.sqrt(phi2 / min(k-1, r-1))
    contingency_coefficient = np.sqrt(chi2 / (chi2 + n))

    resultados.update({
        'Chi2': round(chi2, 4), 'p-valor': round(p, 4), 'Phi^2': round(phi2, 4),
        'Cramer\'s V': round(cramer_v, 4), 'Coeficiente de Contingencia': round(contingency_coefficient, 4)
    })

    if tabla_contingencia.shape == (2, 2):
        tabla_corr = tabla_contingencia + 0.5 if np.any(tabla_contingencia == 0) else tabla_contingencia
        oddsratio_fisher, p_value_fisher = fisher_exact(tabla_corr)

        try:
            table2x2 = sm.stats.Table2x2(tabla_corr.T.values)
            oddsratio = table2x2.oddsratio
            conf_int_or = table2x2.oddsratio_confint()
            riskratio = table2x2.riskratio
            conf_int_rr = table2x2.riskratio_confint()
        except:
            oddsratio, conf_int_or, riskratio, conf_int_rr = np.nan, (np.nan, np.nan), np.nan, (np.nan, np.nan)

        resultados.update({
            'Odds Ratio': round(oddsratio, 4) if not np.isnan(oddsratio) else "No calculable",
            'IC 95% OR': f"({round(conf_int_or[0], 4)}, {round(conf_int_or[1], 4)})" if not np.isnan(oddsratio) else "No aplicable",
            'Razón de Prevalencia': round(riskratio, 4) if not np.isnan(riskratio) else "No calculable",
            'IC 95% RP': f"({round(conf_int_rr[0], 4)}, {round(conf_int_rr[1], 4)})" if not np.isnan(riskratio) else "No aplicable",
            'p-valor Fisher': round(p_value_fisher, 4)
        })

    return resultados

def calcular_estadisticas(df, variable_dependiente, variables_independientes):
    """Calcula estadísticas de asociación para múltiples variables independientes en paralelo."""
    pool = multiprocessing.Pool(processes=min(len(variables_independientes), 4))  # Usa 4 procesos paralelos
    resultados = pool.starmap(calcular_estadisticas_variable, [(df, variable_dependiente, var) for var in variables_independientes])
    pool.close()
    pool.join()

    resultados = [res for res in resultados if res is not None]
    return pd.DataFrame(resultados)

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


import plotly.graph_objects as go

import plotly.graph_objects as go

def graficar_prevalencia_interactiva(df):
    """Genera una gráfica interactiva con doble eje usando Plotly."""
    fig = go.Figure()

    # Agregar barras para el número de casos (eje izquierdo)
    fig.add_trace(go.Bar(
        x=df["Año"], 
        y=df["Personas_año"],
        name="Casos",
        marker=dict(color="deepskyblue"),
        text=df["Personas_año"],
        textposition="outside",
        yaxis="y1"  # Asigna al primer eje Y
    ))

    # Agregar línea para la prevalencia (eje derecho)
    fig.add_trace(go.Scatter(
        x=df["Año"], 
        y=df["Prevalencia (%)"],
        mode="lines+markers",
        name="Prevalencia",
        marker=dict(color="orange", size=8),
        line=dict(width=2),
        text=df["Prevalencia (%)"].astype(str) + "%",
        textposition="top center",
        yaxis="y2"  # Asigna al segundo eje Y
    ))

    # Configuración de ejes
    fig.update_layout(
        title="Sibilancias último año",
        xaxis=dict(title="Año"),
        
        # Eje izquierdo (Casos)
        yaxis=dict(
            title="Número de Casos",
            titlefont=dict(color="deepskyblue"),
            tickfont=dict(color="deepskyblue"),
            side="left"
        ),
        
        # Eje derecho (Prevalencia)
        yaxis2=dict(
            title="Prevalencia (%)",
            titlefont=dict(color="red"),
            tickfont=dict(color="red"),
            overlaying="y",  # Superpone sobre el otro eje
            side="right"
        ),

        legend=dict(x=0.2, y=1.0),
        template="plotly_white"
    )

    st.plotly_chart(fig)




def main():  
    st.title("Asociación y Prevalencia- ARREM Versión 0.1")

    archivo = st.file_uploader("Sube tu archivo Excel", type=["xlsx", "xls"])
    
    if archivo:
        df = cargar_datos(archivo)
        
        if df is not None and not df.empty:
            st.write("Vista previa de los datos:")
            st.write(df.head())

            # 🛠 Secciones con pestañas
            tab1, tab2 = st.tabs(["📊 Análisis de Asociación", "📈 Cálculo de Prevalencia"])

            with tab1:
                st.subheader("Análisis de Asociación Epidemiológica")
                columnas = list(df.columns)

                if "Año" in columnas:
                    años_disponibles = sorted(df["Año"].unique())
                    años_disponibles.insert(0, "Todos")
                    año_seleccionado = st.selectbox("Selecciona el año a analizar:", años_disponibles)
                else:
                    st.error("La columna 'Año' no está en el dataset.")
                    año_seleccionado = "Todos"

                variable_dependiente = st.selectbox("Selecciona la variable dependiente:", columnas)
                variables_independientes = st.multiselect("Selecciona una o más variables independientes:", columnas)

                if st.button("Calcular Estadísticas de Asociación"):
                    if variables_independientes:
                        df_filtrado = df if año_seleccionado == "Todos" else df[df["Año"] == año_seleccionado]

                        if df_filtrado.empty:
                            st.warning(f"No hay datos para el año {año_seleccionado}.")
                        else:
                            resultado_df = calcular_estadisticas(df_filtrado, variable_dependiente, variables_independientes)
                            st.write(f"### **Resultados de la prueba de asociación para {año_seleccionado}:**")
                            st.dataframe(resultado_df)
                    else:
                        st.warning("Por favor, selecciona al menos una variable independiente.")

            #  Aquí comienza la corrección de identación para tab2
            with tab2:
                st.subheader("Cálculo de Prevalencia")
                columnas = list(df.columns)

                if "Año" in columnas:
                    año_col = st.selectbox("Selecciona la columna de Año:", columnas, index=0)
                    sintomas = st.multiselect("Selecciona las columnas de síntomas:", columnas)

                    if sintomas:
                        if st.button("Calcular Prevalencia"):
                            resultado_df = calcular_prevalencia(df, sintomas, año_col)

                            if resultado_df is not None:
                                st.write("### **Resultados de Prevalencia:**")
                                st.dataframe(resultado_df)

                                # 📊 Incluir el gráfico de casos y prevalencia después de mostrar la tabla
                                st.write("### **Gráfico de Casos y Prevalencia**")
                                graficar_prevalencia_interactiva(resultado_df)

                                csv = resultado_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Descargar Resultados en CSV",
                                    data=csv,
                                    file_name="prevalencia_resultados.csv",
                                    mime="text/csv"
                                )
                    else:
                        st.warning("Por favor, selecciona al menos una columna de síntomas.")

if __name__ == "__main__":
    main()