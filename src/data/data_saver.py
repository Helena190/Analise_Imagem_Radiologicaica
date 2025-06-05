import pandas as pd
import os
from src.utils.constants import DATA_DIR, PROCESSED_DATA_FILENAME, DATA_DICTIONARY_FILENAME, FINDING_LABELS_COL, PATIENT_GENDER_COL, VIEW_POSITION_COL, IMAGE_INDEX_COL, PATIENT_AGE_COL, SOURCE_DICTIONARY_FILENAME
from src.utils.logger import logger

class DataSaver:
    """salva dataframes processados e dicionários de dados em arquivos csv."""
    def __init__(self, data_path=DATA_DIR):
        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)
        logger.info(f"datasaver inicializado. os dados serão salvos em: {self.data_path}")

    def save_processed_data(self, df: pd.DataFrame):
        """salva o dataframe processado final em um arquivo csv."""
        filepath = os.path.join(self.data_path, PROCESSED_DATA_FILENAME)
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"dados processados salvos com sucesso em '{filepath}'.")
        except Exception as e:
            logger.error(f"erro ao salvar dados processados em '{filepath}': {e}")

    def save_source_data_dictionary(self):
        """cria e salva o dicionário de dados para os dados de origem originais."""
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
                'string',
                'string',
                'integer',
                'integer',
                'integer',
                'string',
                'string',
                'integer',
                'integer',
                'float',
                'float'
            ],
            'Descrição': [
                'nome do arquivo de imagem radiográfica correspondente a esta entrada de dados.',
                'rótulos indicando a presença de achados clínicos ou doenças identificados na imagem. múltiplos achados para a mesma imagem são separados por "|". o rótulo "no finding" indica que nenhum achado foi identificado.',
                'número sequencial da consulta ou acompanhamento do paciente para a qual esta radiografia foi tirada. o valor 0 geralmente indica a primeira consulta ou exame base.',
                'identificador único para cada paciente no conjunto de dados. permite agrupar múltiplas entradas (radiografias) pertencentes ao mesmo paciente.',
                'idade do paciente em anos no momento da realização da radiografia.',
                'gênero do paciente, indicado como "m" para masculino e "f" para feminino.',
                'posição em que a radiografia foi tirada, indicando a orientação do paciente em relação ao equipamento de raio-x. exemplos comuns incluem "pa" (posteroanterior) e "ap" (anteroposterior).',
                'largura da imagem radiográfica original em pixels.',
                'altura da imagem radiográfica original em pixels.',
                'espaçamento entre pixels no eixo horizontal (x) da imagem original, geralmente medido em milímetros (mm).',
                'espaçamento entre pixels no eixo vertical (y) da imagem original, geralmente medido em milímetros (mm).'
            ]
        }
        df_dictionary = pd.DataFrame(data_dictionary)
        output_filename = os.path.join(self.data_path, SOURCE_DICTIONARY_FILENAME)
        try:
            df_dictionary.to_csv(output_filename, index=False, encoding='utf-8')
            logger.info(f"dicionário de dados de origem criado e salvo em '{output_filename}'.")
            logger.info("\nconteúdo do dicionário de origem:\n" + str(df_dictionary))
        except Exception as e:
            logger.error(f"erro ao salvar dicionário de dados de origem em '{output_filename}': {e}")

    def save_processed_data_dictionary(self, df: pd.DataFrame):
        """cria e salva o dicionário de dados para os dados processados, incluindo descrições para valores codificados."""
        data_dictionary_data = {
            'Nome da Coluna': [],
            'Tipo de Dado': [],
            'Descrição dos Valores (quando aplicável)': []
        }

        for col_name, dtype in df.dtypes.items():
            data_dictionary_data['Nome da Coluna'].append(col_name)
            data_dictionary_data['Tipo de Dado'].append(dtype)

            if col_name == FINDING_LABELS_COL:
                data_dictionary_data['Descrição dos Valores (quando aplicável)'].append("0: sem achado ('no finding'), 1: efusão ('effusion')")
            elif col_name == PATIENT_GENDER_COL:
                data_dictionary_data['Descrição dos Valores (quando aplicável)'].append("0: masculino ('m'), 1: feminino ('f')")
            elif col_name == VIEW_POSITION_COL:
                data_dictionary_data['Descrição dos Valores (quando aplicável)'].append("0: posteroanterior ('pa'), 1: anteroposterior ('ap')")
            else:
                if col_name == IMAGE_INDEX_COL:
                    data_dictionary_data['Descrição dos Valores (quando aplicável)'].append("nome do arquivo de imagem")
                elif col_name == PATIENT_AGE_COL:
                    data_dictionary_data['Descrição dos Valores (quando aplicável)'].append("idade do paciente em anos")
                else:
                    data_dictionary_data['Descrição dos Valores (quando aplicável)'].append("n/a")

        data_dictionary_df = pd.DataFrame(data_dictionary_data)
        filepath = os.path.join(self.data_path, DATA_DICTIONARY_FILENAME)
        try:
            data_dictionary_df.to_csv(filepath, index=False, encoding='utf-8')
            logger.info(f"dicionário de dados processados criado e salvo em '{filepath}'.")
        except Exception as e:
            logger.error(f"erro ao salvar dicionário de dados processados em '{filepath}': {e}")

if __name__ == "__main__":
    logger.info("executando exemplo de datasaver...")
    saver = DataSaver()

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