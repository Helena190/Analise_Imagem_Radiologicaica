import pandas as pd
import os
from src.utils.constants import DATA_DIR, PROCESSED_DATA_FILENAME, DATA_DICTIONARY_FILENAME, FINDING_LABELS_COL, PATIENT_GENDER_COL, VIEW_POSITION_COL, IMAGE_INDEX_COL, PATIENT_AGE_COL, SOURCE_DICTIONARY_FILENAME
from src.utils.logger import logger

class DataSaver:
    """
    Lida com o salvamento de dataframes processados e dicionários de dados em arquivos CSV.
    """
    def __init__(self, data_path=DATA_DIR):
        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)
        logger.info(f"DataSaver initialized. Data will be saved to: {self.data_path}")

    def save_processed_data(self, df: pd.DataFrame):
        """
        Salva o DataFrame processado final em um arquivo CSV.
        """
        filepath = os.path.join(self.data_path, PROCESSED_DATA_FILENAME)
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Processed data saved successfully to '{filepath}'.")
        except Exception as e:
            logger.error(f"Error saving processed data to '{filepath}': {e}")

    def save_source_data_dictionary(self):
        """
        Cria e salva o dicionário de dados para os dados de origem originais.
        """
        data_dictionary = {
            'Nome da Coluna': [
                IMAGE_INDEX_COL,
                FINDING_LABELS_COL,
                'Follow-up #',
                'Patient ID',
                PATIENT_AGE_COL,
                PATIENT_GENDER_COL,
                VIEW_POSITION_COL,
                'OriginalImageWidth',
                'OriginalImageHeight',
                'OriginalImagePixelSpacing_x',
                'OriginalImagePixelSpacing_y'
            ],
            'Tipo de Dado': [
                'string',  # Nome do arquivo
                'string',  # Rótulos de achados (pode conter '|')
                'integer', # Número de acompanhamento
                'integer', # ID do paciente
                'integer', # Idade do paciente
                'string',  # Gênero ('M' ou 'F')
                'string',  # Posição de visualização ('PA', 'AP', etc.)
                'integer', # Largura da imagem
                'integer', # Altura da imagem
                'float',   # Espaçamento pixel X
                'float'    # Espaçamento pixel Y
            ],
            'Descrição': [
                'Nome do arquivo de imagem radiográfica correspondente a esta entrada de dados.',
                'Rótulos indicando a presença de achados clínicos ou doenças identificados na imagem. Múltiplos achados para a mesma imagem são separados por "|". O rótulo "No Finding" indica que nenhum achado foi identificado.',
                'Número sequencial da consulta ou acompanhamento do paciente para a qual esta radiografia foi tirada. O valor 0 geralmente indica a primeira consulta ou exame base.',
                'Identificador único para cada paciente no conjunto de dados. Permite agrupar múltiplas entradas (radiografias) pertencentes ao mesmo paciente.',
                'Idade do paciente em anos no momento da realização da radiografia.',
                'Gênero do paciente, indicado como "M" para Masculino e "F" para Feminino.',
                'Posição em que a radiografia foi tirada, indicando a orientação do paciente em relação ao equipamento de raio-X. Exemplos comuns incluem "PA" (Posteroanterior) e "AP" (Anteroposterior).',
                'Largura da imagem radiográfica original em pixels.',
                'Altura da imagem radiográfica original em pixels.',
                'Espaçamento entre pixels no eixo horizontal (X) da imagem original, geralmente medido em milímetros (mm).',
                'Espaçamento entre pixels no eixo vertical (Y) da imagem original, geralmente medido em milímetros (mm).'
            ]
        }
        df_dictionary = pd.DataFrame(data_dictionary)
        output_filename = os.path.join(self.data_path, SOURCE_DICTIONARY_FILENAME)
        try:
            df_dictionary.to_csv(output_filename, index=False, encoding='utf-8')
            logger.info(f"Source data dictionary created and saved to '{output_filename}'.")
            logger.info("\nContent of source dictionary:\n" + str(df_dictionary))
        except Exception as e:
            logger.error(f"Error saving source data dictionary to '{output_filename}': {e}")

    def save_processed_data_dictionary(self, df: pd.DataFrame):
        """
        Cria e salva o dicionário de dados para os dados processados,
        incluindo descrições para valores codificados.
        """
        data_dictionary_data = {
            'Nome da Coluna': [],
            'Tipo de Dado': [],
            'Descrição dos Valores (quando aplicável)': []
        }

        for col_name, dtype in df.dtypes.items():
            data_dictionary_data['Nome da Coluna'].append(col_name)
            data_dictionary_data['Tipo de Dado'].append(dtype)

            if col_name == FINDING_LABELS_COL:
                data_dictionary_data['Descrição dos Valores (quando aplicável)'].append("0: Sem achado ('No Finding'), 1: Efusão ('Effusion')")
            elif col_name == PATIENT_GENDER_COL:
                data_dictionary_data['Descrição dos Valores (quando aplicável)'].append("0: Masculino ('M'), 1: Feminino ('F')")
            elif col_name == VIEW_POSITION_COL:
                data_dictionary_data['Descrição dos Valores (quando aplicável)'].append("0: Posteroanterior ('PA'), 1: Anteroposterior ('AP')")
            else:
                if col_name == IMAGE_INDEX_COL:
                    data_dictionary_data['Descrição dos Valores (quando aplicável)'].append("Nome do arquivo de imagem")
                elif col_name == PATIENT_AGE_COL:
                    data_dictionary_data['Descrição dos Valores (quando aplicável)'].append("Idade do paciente em anos")
                else:
                    data_dictionary_data['Descrição dos Valores (quando aplicável)'].append("N/A")

        data_dictionary_df = pd.DataFrame(data_dictionary_data)
        filepath = os.path.join(self.data_path, DATA_DICTIONARY_FILENAME)
        try:
            data_dictionary_df.to_csv(filepath, index=False, encoding='utf-8')
            logger.info(f"Processed data dictionary created and saved to '{filepath}'.")
        except Exception as e:
            logger.error(f"Error saving processed data dictionary to '{filepath}': {e}")

if __name__ == "__main__":
    logger.info("Running DataSaver example...")
    saver = DataSaver()

    # Dados processados dummy para teste
    dummy_processed_data = {
        'Image Index': ['img1.png', 'img2.png'],
        'Finding Labels': [0, 1],
        'Patient Age': [25, 35],
        'Patient Gender': [0, 1],
        'View Position': [0, 1],
        'stratify_col': ['0_0', '1_1']
    }
    dummy_processed_df = pd.DataFrame(dummy_processed_data)

    saver.save_processed_data(dummy_processed_df)
    saver.save_source_data_dictionary()
    saver.save_processed_data_dictionary(dummy_processed_df)