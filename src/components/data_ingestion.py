import os
import sys
import pandas as pd
import numpy as np
from lightweight_mmm import utils

from src.exception import CustomException
from src.logger import logging

class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join('artifacts', 'raw_data.csv')

    def initiate_data_ingestion(self):
        logging.info("Iniciando a ingestao de dados")
        try:
            tamanho_real = 156 
            media_data, extra_features, target, costs = utils.simulate_dummy_data(
                data_size=tamanho_real,        
                n_media_channels=3,   
                n_extra_features=2,   
                geos=1                
            )

            # 1. O método .ravel() é mais agressivo que flatten(). Ele esmaga
            # absolutamente qualquer dimensão do JAX para uma linha 1D perfeita.
            target_1d = np.array(target).ravel()
            
            media_np = np.array(media_data).reshape(tamanho_real, -1)
            extra_np = np.array(extra_features).reshape(tamanho_real, -1)

            datas = pd.date_range(end=pd.Timestamp.today().normalize(), periods=tamanho_real, freq='W-MON')
            
            # 2. Monta o Dicionário
            data_dict = {
                'Data': datas,
                'Vendas': target_1d
            }

            canais_midia = ['TV', 'Google_Ads', 'Facebook_Ads']
            for i, canal in enumerate(canais_midia):
                data_dict[f'Investimento_{canal}'] = media_np[:, i].ravel()

            fatores_extras = ['Feriado', 'Sazonalidade']
            for i, fator in enumerate(fatores_extras):
                data_dict[f'Fator_{fator}'] = extra_np[:, i].ravel()

            # DEBUG: Vai imprimir o tamanho de absolutamente TODAS as colunas
            print("--- DEBUG DE TAMANHOS DAS COLUNAS ---")
            for key, val in data_dict.items():
                print(f"Coluna: {key} | Tamanho: {len(val)}")
            print("-------------------------------------")

            # 3. O TRUQUE DE MESTRE: pd.Series()
            # Isso força o Pandas a alinhar os dados automaticamente, ignorando diferenças de tamanho!
            df = pd.DataFrame({chave: pd.Series(valores) for chave, valores in data_dict.items()})
            
            df.set_index('Data', inplace=True)

            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
            df.to_csv(self.raw_data_path)
            
            logging.info("Ingestao concluida com sucesso")
            return df

        except Exception as e:
            logging.error(f"Erro ocorrido na Ingestao: {str(e)}")
            raise CustomException(e, sys)