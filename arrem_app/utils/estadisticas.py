import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
import statsmodels.api as sm
import multiprocessing
from statsmodels.stats.contingency_tables import StratifiedTable

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


import numpy as np
from scipy.stats import chi2

def cochran_armitage_trend_test(casos, controles, exposicion_ordinal):
    """
    Test de tendencia lineal (Mantel-Haenszel extendido / Cochran-Armitage).
    
    Parámetros:
    - casos: lista de casos por nivel (ej. Personas_año)
    - controles: lista de controles por nivel (Total_personas_año - Personas_año)
    - exposicion_ordinal: lista de niveles ordinales (ej. [0,1,2,3,...])

    Retorna:
    - Estadístico Chi²
    - Valor p
    """
    casos = np.array(casos)
    controles = np.array(controles)
    totales = casos + controles
    scores = np.array(exposicion_ordinal)

    N = np.sum(totales)
    T = np.sum(scores * casos)
    P = np.sum(casos) / N

    scores_centered = scores - np.sum(scores * totales) / N
    Var_T = N * P * (1 - P) * np.sum(totales * scores_centered**2) / N

    Z2 = (T - N * P * np.sum(scores * totales) / N) ** 2 / Var_T
    p_value = 1 - chi2.cdf(Z2, df=1)

    return round(Z2, 4), round(p_value, 8)




























