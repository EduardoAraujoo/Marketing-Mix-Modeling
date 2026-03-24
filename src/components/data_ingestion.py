import os
import sys
import pandas as pd
import numpy as np
from lightweight_mmm import utils

from src.exception import CustomException
from src.logger import logging

class DataIngestion:
    def __init__(self):
        # Define onde o dado bruto será salvo
        self.raw_data_path = os.path.join('artifacts', 'raw_data.csv')

    def initiate_data_ingestion(self):
        logging.info("Iniciando o componente de Ingestão de Dados")
        try:
            # 1. Gera os dados do Google
            media_data, extra_features, target, costs = utils.simulate_dummy_data(
                data_size=156,        
                n_media_channels=3,   
                n_extra_features=2,   
                geos=1                
            )

            # 2. Converte JAX Arrays para Numpy
            target_np = np.array(target).flatten()
            media_data_np = np.array(media_data)
            extra_features_np = np.array(extra_features)

            tamanho_real = len(target_np)

            # 3. Normaliza a data (remove as horas) para evitar perda de index no Pandas
            datas = pd.date_range(end=pd.Timestamp.today().normalize(), periods=tamanho_real, freq='W-MON')

            # 4. SOLUÇÃO DEFINITIVA: Criar um dicionário nativo do Python convertendo tudo para listas
            data_dict = {
                'Data': list(datas),
                'Vendas': target_np.tolist()
            }

            canais_midia = ['TV', 'Google_Ads', 'Facebook_Ads']
            for i, canal in enumerate(canais_midia):
                data_dict[f'Investimento_{canal}'] = media_data_np[:, i].tolist()

            fatores_extras = ['Feriado', 'Sazonalidade']
            for i, fator in enumerate(fatores_extras):
                data_dict[f'Fator_{fator}'] = extra_features_np[:, i].tolist()

            # 5. Entrega o dicionário limpo para o Pandas (sem chance de desalinhamento)
            df = pd.DataFrame(data_dict)
            df.set_index('Data', inplace=True)

            # 6. Salva na pasta artifacts
            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
            df.to_csv(self.raw_data_path)

            logging.info("Ingestão de dados concluída com sucesso!")
            return df

        except Exception as e:
            logging.error("Erro ocorrido na Ingestão de Dados")
            raise CustomException(e, sys)
