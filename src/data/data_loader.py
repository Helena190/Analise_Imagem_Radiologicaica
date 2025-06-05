import pandas as pd
import os
import kagglehub
from src.utils.constants import DATA_ENTRY_FILENAME
from src.utils.logger import logger

class DataLoader:
    """carrega dados brutos de caminhos especificados."""
    def __init__(self, data_entry_filename=DATA_ENTRY_FILENAME):
        self.data_entry_filename = data_entry_filename
        self.dataset_path = self._download_dataset()
        self.data_entry_filepath = os.path.join(self.dataset_path, self.data_entry_filename)
        logger.info(f"dataloader inicializado com o caminho do arquivo de entrada de dados: {self.data_entry_filepath}")

    def _download_dataset(self):
        """baixa o conjunto de dados do kagglehub."""
        logger.info("tentando baixar o dataset do kagglehub...")
        try:
            path = kagglehub.dataset_download("nih-chest-xrays/data")
            logger.info(f"dataset baixado com sucesso para: {path}")
            return path
        except Exception as e:
            logger.error(f"ocorreu um erro ao baixar o dataset do kagglehub: {e}")
            return None

    def load_original_data(self):
        """carrega o arquivo csv de entrada de dados original."""
        logger.info(f"tentando carregar dados originais de '{self.data_entry_filepath}'...")
        try:
            de_df = pd.read_csv(self.data_entry_filepath, low_memory=False)
            logger.info(f"carregadas {len(de_df)} entradas de '{self.data_entry_filepath}'.")
            return de_df
        except FileNotFoundError:
            logger.error(f"erro: o arquivo '{self.data_entry_filepath}' não foi encontrado.")
            logger.error("certifique-se de que 'data_entry_2017.csv' está na pasta 'archive'.")
            return None
        except Exception as e:
            logger.error(f"ocorreu um erro ao carregar o arquivo: {e}")
            return None

if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_original_data()

    if df is not None:
        logger.info("\ncabeçalho do dataframe carregado:")
        logger.info(df.head())
        logger.info("\ncolunas do dataframe carregado:")
        logger.info(df.columns)