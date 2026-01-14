import streamlit as st
import pandas as pd
from utils.estadisticas import calcular_estadisticas

MAX_CATEGORIAS = 15  # <-- ajusta: si una col tiene <=15 valores únicos, se filtra como categórica

def _as_str(s: pd.Series) -> pd.Series:
    return s.astype(str).fillna("NA")

def _is_low_cardinality(series: pd.Series, max_cat: int = MAX_CATEGORIAS) -> bool:
    # pocos valores únicos => tratamos como categórica (incluye binarios 0/1)
    return series.dropna().nunique() <= max_cat

def _render_categorical_filter(df: pd.DataFrame, col: str, key_prefix: str) -> dict:
    s = _as_str(df[col])
    uniques = sorted(s.unique().tolist())

    options = ["(Todos)"] + uniques
    default = ["(Todos)"]  # por defecto no filtra

    sel = st.multiselect(
        f"Filtrar **{col}** (elige valores a incluir)",
        options=options,
        default=default,
        key=f"{key_prefix}_{col}_vals"
    )

    # Si eligen (Todos) o no eligen nada => sin filtro
    if not sel or "(Todos)" in sel:
        return {"type": "none"}

    return {"type": "categorical", "values": sel}

def _apply_categorical_filter(df: pd.DataFrame, col: str, selected_values: list[str]) -> pd.DataFrame:
    if not selected_values:
        return df
    s = _as_str(df[col])
    return df[s.isin(selected_values)]

def mostrar_tab_asociacion(df: pd.DataFrame):
    st.subheader("Análisis de Asociación Epidemiológica")
    columnas = list(df.columns)

    # ---- Filtro por año ----
    if "Año" in columnas:
        años_disponibles = sorted(df["Año"].dropna().unique().tolist())
        años_disponibles.insert(0, "Todos")
        año_seleccionado = st.selectbox("Selecciona el año a analizar:", años_disponibles)
    else:
        st.error("La columna 'Año' no está en el dataset.")
        año_seleccionado = "Todos"

    # ---- Selección de variables ----
    variable_dependiente = st.selectbox("Selecciona la variable dependiente:", columnas)
    variables_independientes = st.multiselect(
        "Selecciona variables independientes:",
        [c for c in columnas if c != variable_dependiente]
    )

    # ---- Filtros por variable (SIEMPRE categóricos, por separado) ----
    filtros = {}

    with st.expander("Filtros por variable (categóricos)", expanded=True):
        st.caption(
            "Cada variable tiene su filtro independiente. "
            "Si dejas '(Todos)', no se aplica filtro para esa variable."
        )

        st.markdown("#### Dependiente")
        filtros[variable_dependiente] = _render_categorical_filter(df, variable_dependiente, key_prefix="dep")

        st.markdown("#### Independientes")
        if variables_independientes:
            for col in variables_independientes:
                # si alguna independiente tiene muchísimas categorías, puedes advertir:
                if not _is_low_cardinality(df[col], MAX_CATEGORIAS):
                    st.warning(
                        f"'{col}' tiene muchas categorías ({df[col].dropna().nunique()}). "
                        f"Considera aumentar MAX_CATEGORIAS o no filtrar esa variable."
                    )
                filtros[col] = _render_categorical_filter(df, col, key_prefix="ind")
        else:
            st.info("Selecciona variables independientes para habilitar sus filtros.")

    # ---- Botón de cálculo ----
    if st.button("Calcular Estadísticas de Asociación"):
        if not variables_independientes:
            st.warning("Selecciona al menos una variable independiente.")
            return

        # 1) Filtrar por año
        df_filtrado = df if año_seleccionado == "Todos" else df[df["Año"] == año_seleccionado]
        if df_filtrado.empty:
            st.warning(f"No hay datos para el año {año_seleccionado}.")
            return

        # 2) Aplicar filtros por variable (dep e inds), cada una por separado
        for col, f in filtros.items():
            if f.get("type") == "categorical":
                df_filtrado = _apply_categorical_filter(df_filtrado, col, f.get("values", []))

        if df_filtrado.empty:
            st.warning("Con los filtros seleccionados no quedan registros. Ajusta los filtros.")
            return

        # 3) Calcular
        resultado_df = calcular_estadisticas(df_filtrado, variable_dependiente, variables_independientes)
        st.write(f"### Resultados para {año_seleccionado}:")
        st.dataframe(resultado_df)


