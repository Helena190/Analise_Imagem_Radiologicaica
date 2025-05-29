import pandas as pd
from src.utils.logger import logger
from src.utils.constants import (
    FINDING_LABELS_COL, VIEW_POSITION_COL, PATIENT_GENDER_COL,
    IMAGE_INDEX_COL, PATIENT_AGE_COL, STRATIFY_COL
)

class DataProcessor:
    """
    Lida com o processamento dos dados brutos, incluindo a remoção de colunas,
    filtragem, balanceamento e codificação de características categóricas.
    """
    def __init__(self):
        logger.info("DataProcessor initialized.")

    def drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove colunas que não são necessárias para a análise ou treinamento do modelo.
        """
        columns_to_drop = [
            'Follow-up #',
            'Patient ID',
            'OriginalImage[Width',
            'Height]',
            'OriginalImagePixelSpacing[x',
            'y]',
            'Unnamed: 11'
        ]
        
        # Filtra colunas que não existem no DataFrame
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        
        if existing_columns_to_drop:
            df.drop(columns=existing_columns_to_drop, axis=1, inplace=True)
            logger.info(f"Dropped columns: {existing_columns_to_drop}")
        else:
            logger.warning("No unnecessary columns found to drop based on predefined list.")
        return df

    def analyze_finding_labels(self, df: pd.DataFrame):
        """
        Analisa e registra as contagens de rótulos de achados individuais.
        """
        if FINDING_LABELS_COL in df.columns:
            finding_labels_series = df[FINDING_LABELS_COL]
            split_labels = finding_labels_series.str.split('|')
            all_individual_labels = split_labels.explode()
            disease_counts = all_individual_labels.value_counts()
            logger.info(f"Count of occurrences for each finding label:\n{disease_counts}")
        else:
            logger.warning(f"Column '{FINDING_LABELS_COL}' not found for analysis.")

    def filter_and_balance_data(self, df: pd.DataFrame, target_label: str = 'Effusion') -> pd.DataFrame:
        """
        Filtra o DataFrame para incluir apenas o rótulo alvo e 'No Finding',
        então balanceia o conjunto de dados com base na 'View Position'.
        """
        if FINDING_LABELS_COL not in df.columns or VIEW_POSITION_COL not in df.columns:
            logger.error(f"Required columns '{FINDING_LABELS_COL}' or '{VIEW_POSITION_COL}' not found for filtering and balancing.")
            return df

        selected_df = df[df[FINDING_LABELS_COL].str.contains(target_label, na=False)].copy()
        selected_df[FINDING_LABELS_COL] = target_label
        logger.info(f"Filtered for '{target_label}'. Original dataset size: {len(df)}, '{target_label}' entries: {len(selected_df)}")

        pa_count = (selected_df[VIEW_POSITION_COL] == 'PA').sum()
        ap_count = (selected_df[VIEW_POSITION_COL] == 'AP').sum()
        logger.info(f"Count of entries for '{target_label}' by view position: PA={pa_count}, AP={ap_count}")

        no_finding_df = df[df[FINDING_LABELS_COL] == 'No Finding'].copy()

        pa_no_finding_sample = pd.DataFrame()
        if pa_count > 0:
            pa_no_finding_subset = no_finding_df[no_finding_df[VIEW_POSITION_COL] == 'PA']
            if len(pa_no_finding_subset) >= pa_count:
                pa_no_finding_sample = pa_no_finding_subset.sample(n=pa_count, random_state=42)
                logger.info(f"Sampled {pa_count} 'No Finding' + 'PA' entries.")
            else:
                logger.warning(f"Not enough 'No Finding' + 'PA' entries ({len(pa_no_finding_subset)}) to sample {pa_count}. Selecting all existing.")
                pa_no_finding_sample = pa_no_finding_subset.copy()

        ap_no_finding_sample = pd.DataFrame()
        if ap_count > 0:
            ap_no_finding_subset = no_finding_df[no_finding_df[VIEW_POSITION_COL] == 'AP']
            if len(ap_no_finding_subset) >= ap_count:
                ap_no_finding_sample = ap_no_finding_subset.sample(n=ap_count, random_state=42)
                logger.info(f"Sampled {ap_count} 'No Finding' + 'AP' entries.")
            else:
                logger.warning(f"Not enough 'No Finding' + 'AP' entries ({len(ap_no_finding_subset)}) to sample {ap_count}. Selecting all existing.")
                ap_no_finding_sample = ap_no_finding_subset.copy()

        final_balanced_df = pd.concat([selected_df, pa_no_finding_sample, ap_no_finding_sample], ignore_index=True)
        logger.info(f"Final balanced DataFrame size: {len(final_balanced_df)}")
        logger.info(f"Distribution of '{FINDING_LABELS_COL}' in balanced DataFrame:\n{final_balanced_df[FINDING_LABELS_COL].value_counts()}")
        logger.info(f"Distribution of '{VIEW_POSITION_COL}' in balanced DataFrame:\n{final_balanced_df[VIEW_POSITION_COL].value_counts()}")
        return final_balanced_df

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Codifica características categóricas ('Finding Labels', 'Patient Gender', 'View Position')
        em representações numéricas.
        """
        finding_labels_map = {'No Finding': 0, 'Effusion': 1}
        gender_map = {'M': 0, 'F': 1}
        view_position_map = {'PA': 0, 'AP': 1}

        if FINDING_LABELS_COL in df.columns:
            df[FINDING_LABELS_COL] = df[FINDING_LABELS_COL].map(finding_labels_map)
            logger.info(f"Encoded '{FINDING_LABELS_COL}'.")
        else:
            logger.warning(f"Column '{FINDING_LABELS_COL}' not found for encoding.")

        if PATIENT_GENDER_COL in df.columns:
            df[PATIENT_GENDER_COL] = df[PATIENT_GENDER_COL].map(gender_map)
            logger.info(f"Encoded '{PATIENT_GENDER_COL}'.")
        else:
            logger.warning(f"Column '{PATIENT_GENDER_COL}' not found for encoding.")

        if VIEW_POSITION_COL in df.columns:
            df[VIEW_POSITION_COL] = df[VIEW_POSITION_COL].map(view_position_map)
            logger.info(f"Encoded '{VIEW_POSITION_COL}'.")
        else:
            logger.warning(f"Column '{VIEW_POSITION_COL}' not found for encoding.")
        
        logger.info("Data types after encoding:\n" + str(df.dtypes))
        return df

    def create_stratify_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria uma coluna combinada para estratificação com base nos rótulos de achados e posição de visualização.
        """
        if FINDING_LABELS_COL in df.columns and VIEW_POSITION_COL in df.columns:
            df[STRATIFY_COL] = df[FINDING_LABELS_COL].astype(str) + '_' + df[VIEW_POSITION_COL].astype(str)
            logger.info(f"Created stratification column '{STRATIFY_COL}'.")
        else:
            logger.warning(f"Could not create stratification column. Missing '{FINDING_LABELS_COL}' or '{VIEW_POSITION_COL}'.")
        return df

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Orquestra todo o pipeline de processamento de dados.
        """
        df = self.drop_unnecessary_columns(df)
        self.analyze_finding_labels(df) # Apenas para log, não modifica o df
        df = self.filter_and_balance_data(df)
        df = self.encode_categorical_features(df)
        df = self.create_stratify_column(df)
        return df

if __name__ == "__main__":
    # Exemplo de uso com um DataFrame dummy
    logger.info("Running DataProcessor example...")
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
    processed_df = processor.process_data(dummy_df.copy()) # Usa .copy() para evitar modificar o dummy_df original

    logger.info("\nProcessed DataFrame head:")
    logger.info(processed_df.head())
    logger.info("\nProcessed DataFrame value counts for Finding Labels:")
    logger.info(processed_df[FINDING_LABELS_COL].value_counts())
    logger.info("\nProcessed DataFrame value counts for View Position:")
    logger.info(processed_df[VIEW_POSITION_COL].value_counts())
    logger.info("\nProcessed DataFrame value counts for Patient Gender:")
    logger.info(processed_df[PATIENT_GENDER_COL].value_counts())
    logger.info("\nProcessed DataFrame dtypes:")
    logger.info(processed_df.dtypes)