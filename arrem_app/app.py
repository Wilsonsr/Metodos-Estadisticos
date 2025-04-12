import streamlit as st
from utils.carga_datos import cargar_datos
from componentes.tabs_asociacion import mostrar_tab_asociacion
from componentes.tabs_prevalencia import mostrar_tab_prevalencia

def main():
    st.title("Asociación y Prevalencia - ARREM Versión 0.1")

    archivo = st.file_uploader("Sube tu archivo Excel", type=["xlsx", "xls"])
    
    if archivo:
        df = cargar_datos(archivo)
        
        if df is not None and not df.empty:
            st.write("Vista previa de los datos:")
            st.write(df.head())

            tab1, tab2 = st.tabs(["📊 Análisis de Asociación", "📈 Cálculo de Prevalencia"])
            with tab1:
                mostrar_tab_asociacion(df)
            with tab2:
                mostrar_tab_prevalencia(df)

if __name__ == "__main__":
    main()
