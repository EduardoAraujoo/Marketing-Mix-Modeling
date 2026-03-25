import matplotlib.pyplot as plt
import seaborn as sns
from src.logger import logging
from src.exception import CustomException
import sys

def apply_corporate_style():
    """
    Configura a identidade visual corporativa (Blues/Greys) para o projeto.
    """
    try:
        # Paleta: Navy Blue, Steel Blue, Silver, Slate Gray, Charcoal, Light Blue
        corp_palette = ["#001F3F", "#4682B4", "#C0C0C0", "#708090", "#333333", "#ADD8E6"]
        sns.set_palette(sns.color_palette(corp_palette))
        
        plt.rcParams.update({
            'figure.dpi': 150,
            'figure.figsize': (12, 6),
            'axes.titlesize': 16,
            'axes.titleweight': 'bold',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'font.family': 'sans-serif'
        })
        logging.info("Estilo Visual Corporativo aplicado.")
    except Exception as e:
        raise CustomException(e, sys)