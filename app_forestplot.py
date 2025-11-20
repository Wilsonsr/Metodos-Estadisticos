# app_forestplot.py
# Requiere: pip install streamlit pandas matplotlib forestplot openpyxl

import io

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import forestplot as fp

st.set_page_config(layout="wide")
st.title("Forest plot para regresión logística (OR e IC95%)")

# 1. Cargar archivo
uploaded_file = st.file_uploader(
    "Cargue el archivo de resultados (Excel o CSV)",
    type=["xlsx", "xls", "csv"]
)

if uploaded_file is not None:
    # Leer archivo
    if uploaded_file.name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)

    st.subheader("Vista previa de los datos originales (primeras filas)")
    st.dataframe(df_raw.head(), use_container_width=True)

    # 1b. Elegir desde qué FILA empiezan los datos de la regresión
    st.markdown("### 1. Seleccionar desde qué fila comienzan los resultados de la regresión")
    n_rows = len(df_raw)
    row_start = st.number_input(
        "Índice de fila inicial (0 = primera fila de datos leídos)",
        min_value=0,
        max_value=max(n_rows - 1, 0),
        value=0,
        step=1
    )

    # Tomar desde esa fila hacia abajo
    df_work = df_raw.iloc[row_start:, :].copy().reset_index(drop=True)

    # Usar primera fila como encabezado de columnas
    use_first_row_as_header = st.checkbox(
        "Usar la primera fila de este bloque como nombres de columna",
        value=True
    )

    if use_first_row_as_header and not df_work.empty:
        new_header = df_work.iloc[0]          # fila con los nombres
        df_work = df_work[1:].copy()          # resto de filas son datos
        df_work.columns = new_header          # renombrar columnas
        df_work.reset_index(drop=True, inplace=True)

    st.write("Vista previa de las filas/columnas a usar:")
    st.dataframe(df_work.head(), use_container_width=True)

    # 2. Mapeo de columnas
    st.markdown("### 2. Mapeo de columnas")
    cols = df_work.columns.tolist()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        var_col = st.selectbox("Columna de nombre de variable", cols, index=0 if len(cols) > 0 else None)
    with c2:
        or_col  = st.selectbox("Columna de OR", cols, index=1 if len(cols) > 1 else 0)
    with c3:
        lci_col = st.selectbox("Columna IC95% inferior", cols, index=2 if len(cols) > 2 else 0)
    with c4:
        uci_col = st.selectbox("Columna IC95% superior", cols, index=3 if len(cols) > 3 else 0)
    with c5:
        p_col   = st.selectbox("Columna p-valor", cols, index=4 if len(cols) > 4 else 0)

    # Estandarizar nombres internos
    df = df_work[[var_col, or_col, lci_col, uci_col, p_col]].copy()
    df.columns = ["variable", "OR", "IC95_inf", "IC95_sup", "p_valor"]

    # Convertir a numérico (por si vienen como texto)
    for c in ["OR", "IC95_inf", "IC95_sup", "p_valor"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Opcional: excluir intercepto/constante
    excluir_intercepto = st.checkbox(
        "Excluir término Intercepto/Constante",
        value=True
    )
    if excluir_intercepto:
        df = df[~df["variable"].str.lower().str.contains("intercept|const", na=False)]

    st.markdown("### 3. Editar etiquetas de variables")

    # Columna de etiqueta (para el gráfico) si no existe
    if "Etiqueta" not in df.columns:
        df["Etiqueta"] = df["variable"]

    # Editor interactivo: aquí puedes cambiar nombres de variables o borrar filas
    df_edit = st.data_editor(
        df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Etiqueta": st.column_config.TextColumn("Etiqueta para el gráfico")
        },
        key="tabla_vars"
    )

    # 3b. Preparar datos para el gráfico y tabla
    df_plot = df_edit.copy()
    df_plot = df_plot.dropna(subset=["OR", "IC95_inf", "IC95_sup"])

    if df_plot.empty:
        st.warning("No hay filas válidas para graficar (revisa las columnas numéricas).")
    else:
        # Adaptar nombres para forestplot
        df_fp = df_plot.rename(columns={
            "Etiqueta": "Variable",
            "IC95_inf": "Limite Inferior 95%",
            "IC95_sup": "Limite Superior 95%",
            "p_valor": "Valor p"
        })

        # ---- FORMATO DEL INTERVALO PARA EL GRAFICO ----
        def format_or_ci(row):
            or_str  = f"{row['OR']:.2f}".replace(".", ",")
            lcl_str = f"{row['Limite Inferior 95%']:.2f}".replace(".", ",")
            ucl_str = f"{row['Limite Superior 95%']:.2f}".replace(".", ",")
            # Ejemplo: 1,23 (0,97 - 1,56)
            return f"{or_str} ({lcl_str} - {ucl_str})"

        df_fp["est_ci_custom"] = df_fp.apply(format_or_ci, axis=1)

        # Ordenar por OR
        df_fp = df_fp.sort_values("OR")

        # ==== CONTROLES PARA TAMAÑO DE FIGURA Y FUENTE ====
        st.markdown("### 4. Forest plot (estilo 'forestplot')")
        c_w, c_h, c_fs = st.columns(3)
        with c_w:
            fig_width = st.slider("Ancho de la figura", 5.0, 12.0, 7.0, 0.5)
        with c_h:
            height_per_row = st.slider("Altura por fila", 0.3, 1.0, 0.5, 0.1)
        with c_fs:
            font_size = st.slider("Tamaño de letra", 6, 18, 11, 1)

        fig_height = height_per_row * len(df_fp) + 2

        plt.close("all")

        # ---- GRAFICO TIPO forestplot (sin columna de p-valor) ----
        ax = fp.forestplot(
            df_fp,
            estimate="OR",
            ll="Limite Inferior 95%",
            hl="Limite Superior 95%",
            varlabel="Variable",
            capitalize="capitalize",
            annote=["est_ci_custom"],          # solo nuestra columna custom
            annoteheaders=["OR(IC 95% )"],
            xlabel="Odds ratio",
            table=True,
            figsize=(fig_width, fig_height)
        )

        fig = plt.gcf()
        fig.subplots_adjust(wspace=0.02)

        # ---- AJUSTE GLOBAL DEL TAMAÑO DE LETRA ----
        for ax_ in fig.axes:
            ax_.tick_params(labelsize=font_size)
            for text in ax_.texts:
                text.set_fontsize(font_size)

        # ---- AJUSTE DE RANGO EN X ----
        lcl_min = df_fp["Limite Inferior 95%"].min()
        ucl_max = df_fp["Limite Superior 95%"].max()

        xmin_raw = min(lcl_min, 1)
        xmax_raw = max(ucl_max, 1)
        margin = 0.10 * (xmax_raw - xmin_raw)

        xmin = max(0, xmin_raw - margin)
        xmax = xmax_raw + margin
        plt.xlim(xmin, xmax)

        # Grid y línea roja en 1
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.axvline(x=1, color="red", linestyle="--", linewidth=1)

        st.pyplot(fig, use_container_width=True)

        # 5. Tabla resumen para el informe (mismo formato de intervalo)
        st.markdown("### 5. Tabla para el informe")
        tabla_resumen = df_fp[["Variable", "est_ci_custom", "Valor p"]].copy()
        tabla_resumen.columns = ["Variable", "Est. (95% Conf. Int.)", "Valor p"]
        st.dataframe(tabla_resumen, use_container_width=True)

        # 6. Descargar figura
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)

        st.download_button(
            "Descargar gráfico (PNG)",
            data=buf,
            file_name="forestplot_logistica.png",
            mime="image/png"
        )
