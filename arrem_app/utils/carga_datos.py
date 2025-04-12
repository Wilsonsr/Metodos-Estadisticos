import pandas as pd
import streamlit as st

@st.cache_data
def cargar_datos(archivo):
    """Carga los datos desde un archivo Excel."""
    if archivo is not None:
        return pd.read_excel(archivo)
    return None
