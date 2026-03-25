import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.logger import logging

def get_market_science_metrics(X_data):
    """
    Calcula métricas de rigor estatístico como VIF para o portfólio.
    """
    try:
        vif_df = pd.DataFrame()
        vif_df["Canal"] = X_data.columns
        vif_df["VIF"] = [variance_inflation_factor(X_data.values, i) for i in range(X_data.shape[1])]
        logging.info("VIF calculado para os canais de marketing.")
        return vif_df
    except Exception as e:
        logging.error("Erro no cálculo do VIF.")
        return None