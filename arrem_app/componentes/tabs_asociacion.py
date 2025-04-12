import streamlit as st
from utils.estadisticas import calcular_estadisticas

def mostrar_tab_asociacion(df):
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
    variables_independientes = st.multiselect("Selecciona variables independientes:", columnas)

    if st.button("Calcular Estadísticas de Asociación"):
        if variables_independientes:
            df_filtrado = df if año_seleccionado == "Todos" else df[df["Año"] == año_seleccionado]
            if df_filtrado.empty:
                st.warning(f"No hay datos para el año {año_seleccionado}.")
            else:
                resultado_df = calcular_estadisticas(df_filtrado, variable_dependiente, variables_independientes)
                st.write(f"### Resultados para {año_seleccionado}:")
                st.dataframe(resultado_df)
        else:
            st.warning("Selecciona al menos una variable independiente.")
