# app_final.py
# Requiere: pip install streamlit pandas matplotlib forestplot openpyxl

import io
import re

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import forestplot as fp

st.set_page_config(layout="wide")
st.title("Forest plot para regresión logística (OR e IC95%)")

# -----------------------------
# Utilidades
# -----------------------------
def clean_colname(x):
    x = "" if x is None else str(x)
    x = x.replace("\n", " ").replace("\r", " ").strip()
    x = re.sub(r"\s+", " ", x)
    return x

def make_unique_columns(cols):
    seen = {}
    out = []
    for c in cols:
        c = str(c)
        if c not in seen:
            seen[c] = 1
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
    return out

def normalize_columns(df):
    df.columns = [clean_colname(c) for c in df.columns]
    df.columns = make_unique_columns(df.columns)
    return df

def format_or_ci_coma(or_val, li, ls, nd=2):
    or_str = f"{or_val:.{nd}f}".replace(".", ",")
    li_str = f"{li:.{nd}f}".replace(".", ",")
    ls_str = f"{ls:.{nd}f}".replace(".", ",")
    return f"{or_str} ({li_str} - {ls_str})"


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
        new_header = df_work.iloc[0].tolist()
        df_work = df_work[1:].copy()
        df_work.columns = new_header
        df_work.reset_index(drop=True, inplace=True)

    # Limpieza robusta de nombres de columna (evita casos como OR -> O)
    df_work = normalize_columns(df_work)

    st.write("Vista previa de las filas/columnas a usar:")
    st.dataframe(df_work.head(), use_container_width=True)

    # 2. Mapeo de columnas
    st.markdown("### 2. Mapeo de columnas")
    cols = df_work.columns.tolist()

    # Selector de agrupación (opcional)
    group_options = ["(Sin agrupación)"] + cols
    group_sel = st.selectbox("Columna de grupo (opcional, ej. SINTOMA)", group_options, index=0)
    has_group = group_sel != "(Sin agrupación)"

    # Mapeo principal
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        var_col = st.selectbox("Columna de nombre de variable", cols, index=0 if len(cols) > 0 else None)
    with c2:
        # autoselección si existe exactamente OR
        idx_or = cols.index("OR") if "OR" in cols else (1 if len(cols) > 1 else 0)
        or_col  = st.selectbox("Columna de OR", cols, index=idx_or)
    with c3:
        lci_col = st.selectbox("Columna IC95% inferior", cols, index=2 if len(cols) > 2 else 0)
    with c4:
        uci_col = st.selectbox("Columna IC95% superior", cols, index=3 if len(cols) > 3 else 0)
    with c5:
        p_col_opt = st.selectbox("Columna p-valor (opcional, solo para tabla)", ["(No usar)"] + cols, index=0)
        use_p_in_table = p_col_opt != "(No usar)"

    # Validación temprana: LI y LS no pueden ser la misma columna
    if lci_col == uci_col:
        st.error("Seleccionaste la misma columna para IC95% inferior y superior. "
                 "El superior debe ser diferente (ej. LS95%).")
        st.stop()

    # Estandarizar nombres internos
    base_cols = [var_col, or_col, lci_col, uci_col]
    if has_group:
        base_cols.insert(1, group_sel)
    if use_p_in_table:
        base_cols.append(p_col_opt)

    df = df_work[base_cols].copy()

    if has_group and use_p_in_table:
        df.columns = ["variable", "grupo", "OR", "IC95_inf", "IC95_sup", "p_valor"]
    elif has_group and not use_p_in_table:
        df.columns = ["variable", "grupo", "OR", "IC95_inf", "IC95_sup"]
    elif (not has_group) and use_p_in_table:
        df.columns = ["variable", "OR", "IC95_inf", "IC95_sup", "p_valor"]
    else:
        df.columns = ["variable", "OR", "IC95_inf", "IC95_sup"]

    # Convertir a numérico
    for c in ["OR", "IC95_inf", "IC95_sup"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if use_p_in_table:
        df["p_valor"] = pd.to_numeric(df["p_valor"], errors="coerce")

    if has_group:
        df["grupo"] = df["grupo"].astype(str).fillna("").replace({"nan": ""})

    # Opcional: excluir intercepto/constante
    excluir_intercepto = st.checkbox("Excluir término Intercepto/Constante", value=True)
    if excluir_intercepto:
        df = df[~df["variable"].astype(str).str.lower().str.contains("intercept|const", na=False)]

    st.markdown("### 3. Editar etiquetas de variables")

    if "Etiqueta" not in df.columns:
        df["Etiqueta"] = df["variable"]

    # Editor interactivo
    df_edit = st.data_editor(
        df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={"Etiqueta": st.column_config.TextColumn("Etiqueta para el gráfico")},
        key="tabla_vars"
    )

    # Preparar datos para el gráfico
    df_plot = df_edit.copy()
    df_plot = df_plot.dropna(subset=["OR", "IC95_inf", "IC95_sup"])

    if df_plot.empty:
        st.warning("No hay filas válidas para graficar (revisa las columnas numéricas).")
        st.stop()

    # Adaptar nombres para forestplot
    rename_map = {
        "Etiqueta": "Variable",
        "IC95_inf": "LI 95%",
        "IC95_sup": "LS 95%",
    }
    if has_group:
        rename_map["grupo"] = "Grupo"
    if use_p_in_table:
        rename_map["p_valor"] = "Valor p"

    df_fp = df_plot.rename(columns=rename_map)

    # Corregir ICs invertidos: asegura LI <= LS
    li = df_fp["LI 95%"].copy()
    ls = df_fp["LS 95%"].copy()
    df_fp["LI 95%"] = pd.concat([li, ls], axis=1).min(axis=1)
    df_fp["LS 95%"] = pd.concat([li, ls], axis=1).max(axis=1)

    # Filtrar filas inválidas: OR debe estar entre LI y LS (evita xerr negativo)
    mask_ok = (df_fp["OR"] >= df_fp["LI 95%"]) & (df_fp["OR"] <= df_fp["LS 95%"])
    if (~mask_ok).any():
        st.warning(f"Se excluyeron {(~mask_ok).sum()} filas porque OR no está entre LI y LS (evita error).")
        st.dataframe(df_fp.loc[~mask_ok, ["Variable", "OR", "LI 95%", "LS 95%"]], use_container_width=True)
        df_fp = df_fp.loc[mask_ok].copy()

    if df_fp.empty:
        st.error("Luego de validar ICs, no quedan filas válidas para graficar.")
        st.stop()

    # Columna formateada OR (IC)
    df_fp["est_ci_custom"] = df_fp.apply(
        lambda r: format_or_ci_coma(r["OR"], r["LI 95%"], r["LS 95%"], nd=2),
        axis=1
    )

    # Ordenar (por grupo y OR, o solo OR)
    if has_group:
        df_fp = df_fp.sort_values(["Grupo", "OR"])
    else:
        df_fp = df_fp.sort_values("OR")

    # Controles de tamaño
    st.markdown("### 4. Forest plot (estilo 'forestplot')")
    c_w, c_h, c_fs = st.columns(3)
    with c_w:
        fig_width = st.slider("Ancho de la figura", 5.0, 12.0, 7.0, 0.5)
    with c_h:
        height_per_row = st.slider("Altura por fila", 0.3, 1.2, 0.55, 0.05)
    with c_fs:
        font_size = st.slider("Tamaño de letra", 6, 18, 11, 1)

    fig_height = height_per_row * len(df_fp) + 2
    plt.close("all")

    # ---- Forestplot SIN p-valores en el gráfico (como pediste) ----
    fp_kwargs = dict(
        estimate="OR",
        ll="LI 95%",
        hl="LS 95%",
        varlabel="Variable",
        capitalize="capitalize",
        annote=["est_ci_custom"],
        annoteheaders=["OR (IC 95%)"],
        xlabel="Odds ratio",
        table=True,
        figsize=(fig_width, fig_height)
    )
    if has_group:
        fp_kwargs["groupvar"] = "Grupo"

    ax = fp.forestplot(df_fp, **fp_kwargs)

    fig = plt.gcf()
    fig.subplots_adjust(wspace=0.02)

    # Ajuste global del tamaño de letra
    for ax_ in fig.axes:
        ax_.tick_params(labelsize=font_size)
        for text in ax_.texts:
            text.set_fontsize(font_size)

    # Rango X
    lcl_min = df_fp["LI 95%"].min()
    ucl_max = df_fp["LS 95%"].max()
    xmin_raw = min(lcl_min, 1)
    xmax_raw = max(ucl_max, 1)
    margin = 0.10 * (xmax_raw - xmin_raw) if (xmax_raw - xmin_raw) != 0 else 0.2
    xmin = max(0, xmin_raw - margin)
    xmax = xmax_raw + margin
    ax.set_xlim(xmin, xmax)

    # Grid + línea en OR=1
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.axvline(x=1, color="red", linestyle="--", linewidth=1)

    st.pyplot(fig, use_container_width=True)

    # 5. Tabla para el informe (p-valor opcional SOLO aquí)
    st.markdown("### 5. Tabla para el informe")
    cols_tabla = ["Variable", "est_ci_custom"]
    if has_group:
        cols_tabla = ["Grupo"] + cols_tabla

    if use_p_in_table:
        show_p = st.checkbox("Incluir p-valor en la tabla", value=True)
        if show_p:
            cols_tabla += ["Valor p"]

    tabla_resumen = df_fp[cols_tabla].copy()
    tabla_resumen = tabla_resumen.rename(columns={
        "est_ci_custom": "Est. (IC 95%)",
        "Grupo": "Grupo / Síntoma"
    })
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

