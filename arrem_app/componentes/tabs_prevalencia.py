import streamlit as st
from utils.prevalencia import calcular_prevalencia, graficar_prevalencia_interactiva
from utils.estadisticas import cochran_armitage_trend_test

def mostrar_tab_prevalencia(df):
    st.subheader("Cálculo de Prevalencia")
    columnas = list(df.columns)

    if "Año" in columnas:
        año_col = st.selectbox("Selecciona la columna de Año:", columnas)

        sintomas = st.multiselect("Selecciona uno o más síntomas para analizar:", columnas)

        filtro_columna = st.selectbox("Columna para filtrar (opcional):", ["Ninguno"] + columnas)
        df_filtrado = df.copy()
        categoria_seleccionada = "Bogotá"

        if filtro_columna != "Ninguno" and filtro_columna in df.columns:
            categorias = df[filtro_columna].unique().tolist()
            categorias.insert(0, "Bogotá")
            categoria_seleccionada = st.selectbox(f"Categoría de {filtro_columna}:", categorias)

            if categoria_seleccionada != "Bogotá":
                df_filtrado = df[df[filtro_columna] == categoria_seleccionada]

        if sintomas:
            if st.button("Calcular Prevalencia"):
                resultado_df = calcular_prevalencia(df_filtrado, sintomas, año_col)

                if resultado_df is not None and not resultado_df.empty:
                    st.write(f"### Resultados para {categoria_seleccionada}:")
                    st.dataframe(resultado_df)

                    st.write("### Gráficos y test de tendencia lineal")
                    for sintoma in sintomas:
                        df_sintoma = resultado_df[resultado_df["Síntoma"] == sintoma]
                        if not df_sintoma.empty:
                            st.markdown(f"#### {sintoma}")
                            graficar_prevalencia_interactiva(df_sintoma, sintoma_seleccionado=sintoma)

                            # Cálculo de tendencia
                            casos = df_sintoma["Personas_año"].tolist()
                            controles = (df_sintoma["Total_personas_año"] - df_sintoma["Personas_año"]).tolist()
                            exposicion_ordinal = list(range(len(df_sintoma)))  # Niveles: 0,1,2,3,...

                            chi2_trend, p_trend = cochran_armitage_trend_test(casos, controles, exposicion_ordinal)
                            st.markdown(f"**Chi² tendencia lineal (Mantel-Haenszel extendido):** {chi2_trend} &nbsp;&nbsp;&nbsp; **Valor p:** {p_trend}")
                        else:
                            st.warning(f"No hay datos suficientes para el síntoma: {sintoma}")

                    csv = resultado_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Descargar CSV", data=csv, file_name="prevalencia.csv", mime="text/csv")
                else:
                    st.warning("No se encontraron datos para los síntomas seleccionados.")
        else:
            st.warning("Selecciona al menos un síntoma antes de continuar.")
    else:
        st.error("No se encontró la columna 'Año' en el dataset.")
