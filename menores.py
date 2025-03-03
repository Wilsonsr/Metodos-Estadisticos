import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency, fisher_exact
import statsmodels.api as sm
import multiprocessing

@st.cache_data
def cargar_datos(archivo):
    """Carga los datos desde un archivo Excel."""
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
        'Coeficiente de Contingencia': None, 'Odds Ratio': None, 'IC 95% OR': None,
        'Razón de Prevalencia': None, 'IC 95% RP': None, 'p-valor Fisher': None
    }

    if tabla_contingencia.shape == (2, 2):  # Solo calculamos OR y RP para tablas 2x2
        chi2, p, dof, expected = chi2_contingency(tabla_contingencia)
        oddsratio_fisher, p_value_fisher = fisher_exact(tabla_contingencia)
        phi2 = chi2 / n
        r, k = tabla_contingencia.shape
        cramer_v = np.sqrt(phi2 / min(k-1, r-1))
        contingency_coefficient = np.sqrt(chi2 / (chi2 + n))

        table2x2 = sm.stats.Table2x2(tabla_contingencia.values)
        oddsratio = table2x2.oddsratio
        conf_int_or = table2x2.oddsratio_confint()
        riskratio = table2x2.riskratio
        conf_int_rr = table2x2.riskratio_confint()

        resultados.update({
            'Chi2': round(chi2, 4), 'p-valor': round(p, 4), 'Phi^2': round(phi2, 4),
            'Cramer\'s V': round(cramer_v, 4), 'Coeficiente de Contingencia': round(contingency_coefficient, 4),
            'Odds Ratio': round(oddsratio, 4), 'IC 95% OR': f"({round(conf_int_or[0], 4)}, {round(conf_int_or[1], 4)})",
            'Razón de Prevalencia': round(riskratio, 4), 'IC 95% RP': f"({round(conf_int_rr[0], 4)}, {round(conf_int_rr[1], 4)})",
            'p-valor Fisher': round(p_value_fisher, 4)
        })
    else:
        chi2, p, dof, expected = chi2_contingency(tabla_contingencia)
        phi2 = chi2 / n
        r, k = tabla_contingencia.shape
        cramer_v = np.sqrt(phi2 / min(k-1, r-1))
        contingency_coefficient = np.sqrt(chi2 / (chi2 + n))

        resultados.update({
            'Chi2': round(chi2, 4), 'p-valor': round(p, 4), 'Phi^2': round(phi2, 4),
            'Cramer\'s V': round(cramer_v, 4), 'Coeficiente de Contingencia': round(contingency_coefficient, 4)
        })

    return resultados

def calcular_estadisticas(df, variable_dependiente, variables_independientes):
    """Calcula estadísticas de asociación para múltiples variables independientes en paralelo."""
    pool = multiprocessing.Pool(processes=min(len(variables_independientes), 4))  # Usa 4 procesos paralelos
    resultados = pool.starmap(calcular_estadisticas_variable, [(df, variable_dependiente, var) for var in variables_independientes])
    pool.close()
    pool.join()

    resultados = [res for res in resultados if res is not None]  # Filtrar None
    return pd.DataFrame(resultados)

def main():
    st.title("Análisis de Asociación Epidemiológica con Variables Múltiples")
    
    # 🛠 Se mueve `st.file_uploader()` fuera de `cargar_datos()`
    archivo = st.file_uploader("Sube tu archivo Excel", type=["xlsx", "xls"])
    
    if archivo:
        df = cargar_datos(archivo)  # Pasa el archivo a la función cacheada
        
        if df is not None and not df.empty:
            st.write("Vista previa de los datos:")
            st.write(df.head())

            columnas = list(df.columns)
            variable_dependiente = st.selectbox("Selecciona la variable dependiente:", columnas)
            variables_independientes = st.multiselect("Selecciona una o más variables independientes:", columnas)

            if st.button("Calcular Estadísticas de Asociación"):
                if variables_independientes:
                    resultado_df = calcular_estadisticas(df, variable_dependiente, variables_independientes)
                    st.write("### **Resultados de la prueba de asociación:**")
                    st.dataframe(resultado_df)
                else:
                    st.warning("Por favor, selecciona al menos una variable independiente.")

if __name__ == "__main__":
    main()
