import pandas as pd
from src.utils.logger import logger
from src.utils.constants import (
    FINDING_LABELS_COL, VIEW_POSITION_COL, PATIENT_GENDER_COL,
    IMAGE_INDEX_COL, PATIENT_AGE_COL, STRATIFY_COL
)

class DataProcessor:
    """processa dados brutos: remove colunas, filtra, balanceia e codifica características."""
    def __init__(self):
        logger.info("dataprocessor inicializado.")

    def drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """remove colunas desnecessárias para análise ou treinamento."""
        columns_to_drop = [
            'Follow-up #',
            'Patient ID',
            'OriginalImage[Width',
            'Height]',
            'OriginalImagePixelSpacing[x',
            'y]',
            'Unnamed: 11'
        ]
        
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        
        if existing_columns_to_drop:
            df.drop(columns=existing_columns_to_drop, axis=1, inplace=True)
            logger.info(f"colunas removidas: {existing_columns_to_drop}")
        else:
            logger.warning("nenhuma coluna desnecessária encontrada para remover com base na lista predefinida.")
        return df

    def analyze_finding_labels(self, df: pd.DataFrame):
        """analisa e registra as contagens de rótulos de achados individuais."""
        if FINDING_LABELS_COL in df.columns:
            finding_labels_series = df[FINDING_LABELS_COL]
            split_labels = finding_labels_series.str.split('|')
            all_individual_labels = split_labels.explode()
            disease_counts = all_individual_labels.value_counts()
            logger.info(f"contagem de ocorrências para cada rótulo de achado:\n{disease_counts}")
        else:
            logger.warning(f"coluna '{FINDING_LABELS_COL}' não encontrada para análise.")

    def filter_and_balance_data(self, df: pd.DataFrame, target_label: str = 'Effusion') -> pd.DataFrame:
        """filtra o dataframe para incluir apenas o rótulo alvo e 'no finding', então balanceia o conjunto de dados com base na 'view position'."""
        if FINDING_LABELS_COL not in df.columns or VIEW_POSITION_COL not in df.columns:
            logger.error(f"colunas necessárias '{FINDING_LABELS_COL}' ou '{VIEW_POSITION_COL}' não encontradas para filtragem e balanceamento.")
            return df

        selected_df = df[df[FINDING_LABELS_COL].str.contains(target_label, na=False)].copy()
        selected_df[FINDING_LABELS_COL] = target_label
        logger.info(f"filtrado para '{target_label}'. tamanho do dataset original: {len(df)}, entradas de '{target_label}': {len(selected_df)}")

        pa_count = (selected_df[VIEW_POSITION_COL] == 'PA').sum()
        ap_count = (selected_df[VIEW_POSITION_COL] == 'AP').sum()
        logger.info(f"contagem de entradas para '{target_label}' por posição de visualização: pa={pa_count}, ap={ap_count}")

        no_finding_df = df[df[FINDING_LABELS_COL] == 'No Finding'].copy()

        pa_no_finding_sample = pd.DataFrame()
        if pa_count > 0:
            pa_no_finding_subset = no_finding_df[no_finding_df[VIEW_POSITION_COL] == 'PA']
            if len(pa_no_finding_subset) >= pa_count:
                pa_no_finding_sample = pa_no_finding_subset.sample(n=pa_count, random_state=42)
                logger.info(f"amostradas {pa_count} entradas 'no finding' + 'pa'.")
            else:
                logger.warning(f"não há entradas 'no finding' + 'pa' suficientes ({len(pa_no_finding_subset)}) para amostrar {pa_count}. selecionando todas as existentes.")
                pa_no_finding_sample = pa_no_finding_subset.copy()

        ap_no_finding_sample = pd.DataFrame()
        if ap_count > 0:
            ap_no_finding_subset = no_finding_df[no_finding_df[VIEW_POSITION_COL] == 'AP']
            if len(ap_no_finding_subset) >= ap_count:
                ap_no_finding_sample = ap_no_finding_subset.sample(n=ap_count, random_state=42)
                logger.info(f"amostradas {ap_count} entradas 'no finding' + 'ap'.")
            else:
                logger.warning(f"não há entradas 'no finding' + 'ap' suficientes ({len(ap_no_finding_subset)}) para amostrar {ap_count}. selecionando todas as existentes.")
                ap_no_finding_sample = ap_no_finding_subset.copy()

        final_balanced_df = pd.concat([selected_df, pa_no_finding_sample, ap_no_finding_sample], ignore_index=True)
        logger.info(f"tamanho final do dataframe balanceado: {len(final_balanced_df)}")
        logger.info(f"distribuição de '{FINDING_LABELS_COL}' no dataframe balanceado:\n{final_balanced_df[FINDING_LABELS_COL].value_counts()}")
        logger.info(f"distribuição de '{VIEW_POSITION_COL}' no dataframe balanceado:\n{final_balanced_df[VIEW_POSITION_COL].value_counts()}")
        return final_balanced_df

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """codifica características categóricas ('finding labels', 'patient gender', 'view position') em representações numéricas."""
        finding_labels_map = {'No Finding': 0, 'Effusion': 1}
        gender_map = {'M': 0, 'F': 1}
        view_position_map = {'PA': 0, 'AP': 1}

        if FINDING_LABELS_COL in df.columns:
            df[FINDING_LABELS_COL] = df[FINDING_LABELS_COL].map(finding_labels_map)
            logger.info(f"codificado '{FINDING_LABELS_COL}'.")
        else:
            logger.warning(f"coluna '{FINDING_LABELS_COL}' não encontrada para codificação.")

        if PATIENT_GENDER_COL in df.columns:
            df[PATIENT_GENDER_COL] = df[PATIENT_GENDER_COL].map(gender_map)
            logger.info(f"codificado '{PATIENT_GENDER_COL}'.")
        else:
            logger.warning(f"coluna '{PATIENT_GENDER_COL}' não encontrada para codificação.")

        if VIEW_POSITION_COL in df.columns:
            df[VIEW_POSITION_COL] = df[VIEW_POSITION_COL].map(view_position_map)
            logger.info(f"codificado '{VIEW_POSITION_COL}'.")
        else:
            logger.warning(f"coluna '{VIEW_POSITION_COL}' não encontrada para codificação.")
        
        logger.info("tipos de dados após a codificação:\n" + str(df.dtypes))
        return df

    def create_stratify_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """cria uma coluna combinada para estratificação com base nos rótulos de achados e posição de visualização."""
        if FINDING_LABELS_COL in df.columns and VIEW_POSITION_COL in df.columns:
            df[STRATIFY_COL] = df[FINDING_LABELS_COL].astype(str) + '_' + df[VIEW_POSITION_COL].astype(str)
            logger.info(f"coluna de estratificação '{STRATIFY_COL}' criada.")
        else:
            logger.warning(f"não foi possível criar a coluna de estratificação. faltando '{FINDING_LABELS_COL}' ou '{VIEW_POSITION_COL}'.")
        return df

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """orquestra todo o pipeline de processamento de dados."""
        df = self.drop_unnecessary_columns(df)
        self.analyze_finding_labels(df) # apenas para log, não modifica o df
        df = self.filter_and_balance_data(df)
        df = self.encode_categorical_features(df)
        df = self.create_stratify_column(df)
        return df

if __name__ == "__main__":
    logger.info("executando exemplo de dataprocessor...")
    dummy_data = {
        'Image Index': ['img1.png', 'img2.png', 'img3.png', 'img4.png', 'img5.png', 'img6.png'],
        'Finding Labels': ['No Finding', 'Effusion', 'Effusion|Cardiomegaly', 'No Finding', 'Effusion', 'No Finding'],
        'Follow-up #': [0, 1, 2, 0, 1, 0],
        'Patient ID': [1, 2, 3, 4, 5, 6],
        'Patient Age': [25, 35, 45, 55, 65, 75],
        'Patient Gender': ['M', 'F', 'M', 'F', 'M', 'F'],
        'View Position': ['PA', 'AP', 'PA', 'AP', 'PA', 'AP'],
        'OriginalImage[Width': [1024, 1024, 1024, 1024, 1024, 1024],
        'Height]': [1024, 1024, 1024, 1024, 1024, 1024],
        'OriginalImagePixelSpacing[x': [0.143, 0.143, 0.143, 0.143, 0.143, 0.143],
        'y]': [0.143, 0.143, 0.143, 0.143, 0.143, 0.143],
        'Unnamed: 11': [None, None, None, None, None, None]
    }
    dummy_df = pd.DataFrame(dummy_data)

    processor = DataProcessor()
    processed_df = processor.process_data(dummy_df.copy()) # usa .copy() para evitar modificar o dummy_df original

    logger.info("\ncabeçalho do dataframe processado:")
    logger.info(processed_df.head())
    logger.info("\ncontagens de valores do dataframe processado para finding labels:")
    logger.info(processed_df[FINDING_LABELS_COL].value_counts())
    logger.info("\ncontagens de valores do dataframe processado para view position:")
    logger.info(processed_df[VIEW_POSITION_COL].value_counts())
    logger.info("\ncontagens de valores do dataframe processado para patient gender:")
    logger.info(processed_df[PATIENT_GENDER_COL].value_counts())
    logger.info("\ntipos de dados do dataframe processado:")
    logger.info(processed_df.dtypes)