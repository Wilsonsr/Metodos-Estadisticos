import streamlit as st
import pandas as pd
import scipy.stats as stats

def cargar_datos():
    """Función para cargar los datos desde un archivo Excel"""
    archivo = st.file_uploader("Sube tu archivo Excel", type=["xlsx", "xls"])
    if archivo is not None:
        df = pd.read_excel(archivo)
        return df
    return None

def chi_cuadrado(df, variable_dependiente, variable_independiente):
    """Función para calcular la prueba de chi-cuadrado entre dos variables categóricas"""
    tabla = pd.crosstab(df[variable_dependiente], df[variable_independiente])
    chi2, p, dof, expected = stats.chi2_contingency(tabla)
    return chi2, p

def main():
    st.title("Análisis de Asociación Epidemiológica")
    st.write("Este aplicativo permite seleccionar variables de una base de datos y calcular la prueba de chi-cuadrado.")
    
    df = cargar_datos()
    
    if df is not None:
        st.write("Vista previa de los datos:")
        st.write(df.head())
        
        # Selección de variables
        columnas = list(df.columns)
        variable_dependiente = st.selectbox("Selecciona la variable dependiente:", columnas)
        variable_independiente = st.selectbox("Selecciona la variable independiente:", columnas)
        
        if st.button("Calcular Chi-cuadrado"):
            chi2, p = chi_cuadrado(df, variable_dependiente, variable_independiente)
            st.write(f"**Resultado de la prueba de Chi-cuadrado:**")
            st.write(f"Chi-cuadrado: {chi2:.4f}")
            st.write(f"p-valor: {p:.4f}")
            
            if p < 0.05:
                st.success("Existe una asociación significativa entre las variables seleccionadas (p < 0.05).")
            else:
                st.warning("No se encontró una asociación significativa entre las variables seleccionadas (p ≥ 0.05).")

if __name__ == "__main__":
    main()
