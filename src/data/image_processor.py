import pandas as pd
import os
import shutil
import glob
from sklearn.model_selection import train_test_split
from PIL import Image
from src.utils.logger import logger
from src.utils.constants import (
    DATA_DIR, PROCESSED_DATA_FILENAME,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    TARGET_WIDTH, TARGET_HEIGHT,
    VIEW_POSITION_DIR_MAP, FINDING_LABELS_DIR_MAP,
    IMAGE_INDEX_COL, SOURCE_PATH_COL, FINDING_LABELS_COL, VIEW_POSITION_COL, STRATIFY_COL
)

class ImageProcessor:
    """
    Lida com a divisão de dados, criação de diretórios de imagem,
    e o redimensionamento e cópia de imagens para o conjunto de dados.
    """
    def __init__(self, data_path=DATA_DIR):
        from src.data.data_loader import DataLoader
        self.data_loader = DataLoader()
        self.data_path = data_path
        self.processed_data_filepath = os.path.join(self.data_path, PROCESSED_DATA_FILENAME)
        logger.info(f"ImageProcessor initialized. Processed data expected at: {self.processed_data_filepath}")

    def load_processed_data(self) -> pd.DataFrame:
        """
        Carrega o arquivo CSV de entrada de dados processados.
        """
        logger.info(f"Loading processed data from '{self.processed_data_filepath}'...")
        if not os.path.exists(self.processed_data_filepath):
            logger.error(f"Error: Processed data file not found at '{self.processed_data_filepath}'.")
            logger.error("Please ensure data processing has been completed to generate this file.")
            return None
        try:
            processed_df = pd.read_csv(self.processed_data_filepath)
            logger.info(f"Loaded {len(processed_df)} processed entries.")
            logger.info(f"Columns: {processed_df.columns.tolist()}")
            logger.info(f"Value counts for '{FINDING_LABELS_COL}':\n{processed_df[FINDING_LABELS_COL].value_counts()}")
            logger.info(f"Value counts for '{VIEW_POSITION_COL}':\n{processed_df[VIEW_POSITION_COL].value_counts()}")
            return processed_df
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            return None

    def build_image_path_map(self) -> dict:
        """
        Escaneia os diretórios do arquivo para construir um mapa do índice da imagem para o caminho da fonte original.
        """
        logger.info("Building image path map from downloaded dataset...")
        image_path_map = {}
        dataset_base_path = self.data_loader.dataset_path
        if dataset_base_path is None:
            logger.error("Dataset path not available. Cannot build image path map.")
            return {}

        # Escaneia os diretórios images_001 a images_012 dentro do conjunto de dados baixado
        for i in range(1, 13):
            image_dir = os.path.join(dataset_base_path, f'images_{i:03d}', 'images')
            if os.path.exists(image_dir):
                files_in_dir = glob.glob(os.path.join(image_dir, '*.png'))
                for f in files_in_dir:
                    image_path_map[os.path.basename(f)] = f
            else:
                logger.warning(f"Directory '{image_dir}' not found within dataset. Ignoring.")
        logger.info(f"Scanning finished. Total unique image files found in dataset: {len(image_path_map)}")
        return image_path_map

    def add_source_path_to_df(self, df: pd.DataFrame, image_path_map: dict) -> pd.DataFrame:
        """
        Adiciona o caminho da fonte de cada imagem ao DataFrame e lida com imagens ausentes.
        """
        if IMAGE_INDEX_COL not in df.columns:
            logger.error(f"Column '{IMAGE_INDEX_COL}' not found in DataFrame. Cannot add source paths.")
            return df

        df[SOURCE_PATH_COL] = df[IMAGE_INDEX_COL].map(image_path_map)
        missing_images = df[df[SOURCE_PATH_COL].isnull()]
        if not missing_images.empty:
            logger.warning(f"{len(missing_images)} images listed in processed data were not found in the archive.")
            logger.warning(f"First 10 missing image indices: {missing_images[IMAGE_INDEX_COL].tolist()[:10]} ...")
            df.dropna(subset=[SOURCE_PATH_COL], inplace=True)
            logger.info(f"Removed {len(missing_images)} entries due to missing image files. Remaining entries: {len(df)}")
        return df

    def split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divide o DataFrame em conjuntos de treinamento, validação e teste com estratificação.
        """
        logger.info("Splitting data into training, validation, and test sets...")
        if STRATIFY_COL not in df.columns:
            logger.error(f"Stratification column '{STRATIFY_COL}' not found. Cannot perform stratified split.")
            logger.info("Tentando divisão não estratificada. Isso pode levar a conjuntos de dados desbalanceados.")
            # Fallback to non-stratified split if stratify_col is missing
            train_df, temp_df = train_test_split(
                df,
                test_size=(VAL_RATIO + TEST_RATIO),
                random_state=42
            )
            relative_test_size = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
            val_df, test_df = train_test_split(
                temp_df,
                test_size=relative_test_size,
                random_state=42
            )
        else:
            train_df, temp_df = train_test_split(
                df,
                test_size=(VAL_RATIO + TEST_RATIO),
                random_state=42,
                stratify=df[STRATIFY_COL]
            )
            relative_test_size = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
            val_df, test_df = train_test_split(
                temp_df,
                test_size=relative_test_size,
                random_state=42,
                stratify=temp_df[STRATIFY_COL]
            )
            # Remove a coluna de estratificação temporária
            train_df = train_df.drop(columns=STRATIFY_COL)
            val_df = val_df.drop(columns=STRATIFY_COL)
            test_df = test_df.drop(columns=STRATIFY_COL)

        logger.info(f"Train set size: {len(train_df)}")
        logger.info(f"Validation set size: {len(val_df)}")
        logger.info(f"Test set size: {len(test_df)}")
        
        logger.info(f"Distribution of '{FINDING_LABELS_COL}' in train set:\n{train_df[FINDING_LABELS_COL].value_counts(normalize=True)}")
        logger.info(f"Distribution of '{FINDING_LABELS_COL}' in validation set:\n{val_df[FINDING_LABELS_COL].value_counts(normalize=True)}")
        logger.info(f"Distribution of '{FINDING_LABELS_COL}' in test set:\n{test_df[FINDING_LABELS_COL].value_counts(normalize=True)}")
        
        return train_df, val_df, test_df

    def create_destination_directories(self):
        """
        Cria a estrutura de diretórios necessária para armazenar imagens divididas e processadas.
        """
        logger.info("Creating destination directories...")
        split_names = ['train', 'validation', 'test']
        for split_name in split_names:
            for view_pos_val, view_pos_str in VIEW_POSITION_DIR_MAP.items():
                for finding_label_val, finding_label_str in FINDING_LABELS_DIR_MAP.items():
                    dest_dir = os.path.join(self.data_path, split_name, view_pos_str, finding_label_str)
                    os.makedirs(dest_dir, exist_ok=True)
                    # logger.debug(f"Diretório criado/garantido: {dest_dir}") # Muito verboso para log geral
        logger.info("Destination directory structure created.")

    def resize_and_copy_images(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Redimensiona e copia imagens para seus respectivos diretórios de destino.
        """
        logger.info(f"Resizing and copying images to new structure ({TARGET_WIDTH}x{TARGET_HEIGHT} pixels)...")
        logger.info("This process may take some time depending on the number of images.")

        split_dfs = {'train': train_df, 'validation': val_df, 'test': test_df}
        processed_count = 0
        total_files_to_process = sum(len(df) for df in split_dfs.values())

        for split_name, df in split_dfs.items():
            logger.info(f"Processing {len(df)} files for split: {split_name}")
            for index, row in df.iterrows():
                image_index = row[IMAGE_INDEX_COL]
                source_path = row[SOURCE_PATH_COL]
                finding_label_val = row[FINDING_LABELS_COL]
                view_pos_val = row[VIEW_POSITION_COL]

                view_pos_str = VIEW_POSITION_DIR_MAP[view_pos_val]
                finding_label_str = FINDING_LABELS_DIR_MAP[finding_label_val]

                dest_dir = os.path.join(self.data_path, split_name, view_pos_str, finding_label_str)
                dest_path = os.path.join(dest_dir, image_index)

                try:
                    with Image.open(source_path) as img:
                        resized_img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
                        resized_img.save(dest_path, format='PNG')
                    processed_count += 1
                    if processed_count % 500 == 0:
                        logger.info(f"  Processed {processed_count}/{total_files_to_process} files...")
                except FileNotFoundError:
                    logger.error(f"  Error: Source file not found for {image_index} at {source_path}. Skipping.")
                except Exception as e:
                    logger.error(f"  Error processing {image_index} from {source_path} to {dest_path}: {e}")
        
        logger.info(f"Image processing and copying complete. Total files processed: {processed_count}")
        logger.info("Image processing finished.")

    def process_images_pipeline(self):
        """
        Orquestra todo o pipeline de processamento de imagens.
        """
        processed_df = self.load_processed_data()
        if processed_df is None:
            logger.error("Image processing aborted due to failure in loading processed data.")
            return

        image_path_map = self.build_image_path_map()
        processed_df = self.add_source_path_to_df(processed_df, image_path_map)

        if processed_df.empty:
            logger.error("No images to process after filtering for missing source paths. Aborting image processing.")
            return

        train_df, val_df, test_df = self.split_data(processed_df)
        self.create_destination_directories()
        self.resize_and_copy_images(train_df, val_df, test_df)

if __name__ == "__main__":
    logger.info("Running ImageProcessor example...")
    # Para um teste completo, você precisaria de um diretório 'data' dummy com 'data_entry.csv'
    
    # Cria um dummy processed_data.csv para teste
    test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
    os.makedirs(test_data_dir, exist_ok=True)
    test_processed_data_filepath = os.path.join(test_data_dir, PROCESSED_DATA_FILENAME)

    if not os.path.exists(test_processed_data_filepath):
        logger.info(f"Creating a dummy '{PROCESSED_DATA_FILENAME}' for testing ImageProcessor...")
        dummy_processed_data = {
            IMAGE_INDEX_COL: [f'000000{i:02d}_000.png' for i in range(1, 101)],
            FINDING_LABELS_COL: [i % 2 for i in range(100)], # Alternating 0 and 1
            VIEW_POSITION_COL: [i % 2 for i in range(100)], # Alternating 0 and 1
            STRATIFY_COL: [f'{i % 2}_{i % 2}' for i in range(100)]
        }
        pd.DataFrame(dummy_processed_data).to_csv(test_processed_data_filepath, index=False)
        logger.info("Dummy processed data file created.")

    processor = ImageProcessor(data_path=test_data_dir)
    processor.process_images_pipeline()

    # Limpa arquivos e diretórios dummy (opcional, descomente para limpeza real)
    # shutil.rmtree(test_data_dir)