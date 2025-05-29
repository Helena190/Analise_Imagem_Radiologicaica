import pandas as pd
import os
import kagglehub
from src.utils.constants import DATA_ENTRY_FILENAME
from src.utils.logger import logger

class DataLoader:
    """
    Lida com o carregamento de dados brutos de caminhos especificados.
    """
    def __init__(self, data_entry_filename=DATA_ENTRY_FILENAME):
        self.data_entry_filename = data_entry_filename
        self.dataset_path = self._download_dataset()
        self.data_entry_filepath = os.path.join(self.dataset_path, self.data_entry_filename)
        logger.info(f"DataLoader initialized with data entry filepath: {self.data_entry_filepath}")

    def _download_dataset(self):
        """
        Baixa o conjunto de dados do Kagglehub.
        """
        logger.info("Attempting to download dataset from Kagglehub...")
        try:
            path = kagglehub.dataset_download("nih-chest-xrays/data")
            logger.info(f"Successfully downloaded dataset to: {path}")
            return path
        except Exception as e:
            logger.error(f"An error occurred while downloading the dataset from Kagglehub: {e}")
            return None

    def load_original_data(self):
        """
        Carrega o arquivo CSV de entrada de dados original.

        Retorna:
            pd.DataFrame: O DataFrame carregado, ou None se ocorrer um erro.
        """
        logger.info(f"Attempting to load original data from '{self.data_entry_filepath}'...")
        try:
            de_df = pd.read_csv(self.data_entry_filepath, low_memory=False)
            logger.info(f"Successfully loaded {len(de_df)} entries from '{self.data_entry_filepath}'.")
            return de_df
        except FileNotFoundError:
            logger.error(f"Error: The file '{self.data_entry_filepath}' was not found.")
            logger.error("Please ensure 'Data_Entry_2017.csv' is in the 'archive' folder.")
            return None
        except Exception as e:
            logger.error(f"An error occurred while loading the file: {e}")
            return None

if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_original_data()

    if df is not None:
        logger.info("\nLoaded DataFrame head:")
        logger.info(df.head())
        logger.info("\nLoaded DataFrame columns:")
        logger.info(df.columns)