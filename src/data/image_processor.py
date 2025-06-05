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
    """divide dados, cria diretórios de imagem, redimensiona e copia imagens."""
    def __init__(self, data_path=DATA_DIR):
        from src.data.data_loader import DataLoader
        self.data_loader = DataLoader()
        self.data_path = data_path
        self.processed_data_filepath = os.path.join(self.data_path, PROCESSED_DATA_FILENAME)
        logger.info(f"imageprocessor inicializado. dados processados esperados em: {self.processed_data_filepath}")

    def load_processed_data(self) -> pd.DataFrame:
        """carrega o arquivo csv de entrada de dados processados."""
        logger.info(f"carregando dados processados de '{self.processed_data_filepath}'...")
        if not os.path.exists(self.processed_data_filepath):
            logger.error(f"erro: arquivo de dados processados não encontrado em '{self.processed_data_filepath}'.")
            logger.error("certifique-se de que o processamento de dados foi concluído para gerar este arquivo.")
            return None
        try:
            processed_df = pd.read_csv(self.processed_data_filepath)
            logger.info(f"carregadas {len(processed_df)} entradas processadas.")
            logger.info(f"colunas: {processed_df.columns.tolist()}")
            logger.info(f"contagens de valores para '{FINDING_LABELS_COL}':\n{processed_df[FINDING_LABELS_COL].value_counts()}")
            logger.info(f"contagens de valores para '{VIEW_POSITION_COL}':\n{processed_df[VIEW_POSITION_COL].value_counts()}")
            return processed_df
        except Exception as e:
            logger.error(f"erro ao carregar dados processados: {e}")
            return None

    def build_image_path_map(self) -> dict:
        """escaneia os diretórios do arquivo para construir um mapa do índice da imagem para o caminho da fonte original."""
        logger.info("construindo mapa de caminho de imagem do dataset baixado...")
        image_path_map = {}
        dataset_base_path = self.data_loader.dataset_path
        if dataset_base_path is None:
            logger.error("caminho do dataset não disponível. não é possível construir o mapa de caminho da imagem.")
            return {}

        # escaneia os diretórios images_001 a images_012 dentro do conjunto de dados baixado
        for i in range(1, 13):
            image_dir = os.path.join(dataset_base_path, f'images_{i:03d}', 'images')
            if os.path.exists(image_dir):
                files_in_dir = glob.glob(os.path.join(image_dir, '*.png'))
                for f in files_in_dir:
                    image_path_map[os.path.basename(f)] = f
            else:
                logger.warning(f"diretório '{image_dir}' não encontrado no dataset. ignorando.")
        logger.info(f"escaneamento concluído. total de arquivos de imagem únicos encontrados no dataset: {len(image_path_map)}")
        return image_path_map

    def add_source_path_to_df(self, df: pd.DataFrame, image_path_map: dict) -> pd.DataFrame:
        """adiciona o caminho da fonte de cada imagem ao dataframe e lida com imagens ausentes."""
        if IMAGE_INDEX_COL not in df.columns:
            logger.error(f"coluna '{IMAGE_INDEX_COL}' não encontrada no dataframe. não é possível adicionar caminhos de origem.")
            return df

        df[SOURCE_PATH_COL] = df[IMAGE_INDEX_COL].map(image_path_map)
        missing_images = df[df[SOURCE_PATH_COL].isnull()]
        if not missing_images.empty:
            logger.warning(f"{len(missing_images)} imagens listadas nos dados processados não foram encontradas no arquivo.")
            logger.warning(f"primeiros 10 índices de imagens ausentes: {missing_images[IMAGE_INDEX_COL].tolist()[:10]} ...")
            df.dropna(subset=[SOURCE_PATH_COL], inplace=True)
            logger.info(f"removidas {len(missing_images)} entradas devido a arquivos de imagem ausentes. entradas restantes: {len(df)}")
        return df

    def split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """divide o dataframe em conjuntos de treinamento, validação e teste com estratificação."""
        logger.info("dividindo dados em conjuntos de treinamento, validação e teste...")
        if STRATIFY_COL not in df.columns:
            logger.error(f"coluna de estratificação '{STRATIFY_COL}' não encontrada. não é possível realizar a divisão estratificada.")
            logger.info("tentando divisão não estratificada. isso pode levar a conjuntos de dados desbalanceados.")
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
            train_df = train_df.drop(columns=STRATIFY_COL)
            val_df = val_df.drop(columns=STRATIFY_COL)
            test_df = test_df.drop(columns=STRATIFY_COL)

        logger.info(f"tamanho do conjunto de treino: {len(train_df)}")
        logger.info(f"tamanho do conjunto de validação: {len(val_df)}")
        logger.info(f"tamanho do conjunto de teste: {len(test_df)}")
        
        logger.info(f"distribuição de '{FINDING_LABELS_COL}' no conjunto de treino:\n{train_df[FINDING_LABELS_COL].value_counts(normalize=True)}")
        logger.info(f"distribuição de '{FINDING_LABELS_COL}' no conjunto de validação:\n{val_df[FINDING_LABELS_COL].value_counts(normalize=True)}")
        logger.info(f"distribuição de '{FINDING_LABELS_COL}' no conjunto de teste:\n{test_df[FINDING_LABELS_COL].value_counts(normalize=True)}")
        
        return train_df, val_df, test_df

    def create_destination_directories(self):
        """cria a estrutura de diretórios para armazenar imagens divididas e processadas."""
        logger.info("criando diretórios de destino...")
        split_names = ['train', 'validation', 'test']
        for split_name in split_names:
            for view_pos_val, view_pos_str in VIEW_POSITION_DIR_MAP.items():
                for finding_label_val, finding_label_str in FINDING_LABELS_DIR_MAP.items():
                    dest_dir = os.path.join(self.data_path, split_name, view_pos_str, finding_label_str)
                    os.makedirs(dest_dir, exist_ok=True)
        logger.info("estrutura de diretórios de destino criada.")

    def resize_and_copy_images(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """redimensiona e copia imagens para seus respectivos diretórios de destino."""
        logger.info(f"redimensionando e copiando imagens para nova estrutura ({TARGET_WIDTH}x{TARGET_HEIGHT} pixels)...")
        logger.info("este processo pode levar algum tempo dependendo do número de imagens.")

        split_dfs = {'train': train_df, 'validation': val_df, 'test': test_df}
        processed_count = 0
        total_files_to_process = sum(len(df) for df in split_dfs.values())

        for split_name, df in split_dfs.items():
            logger.info(f"processando {len(df)} arquivos para a divisão: {split_name}")
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
                        logger.info(f"  processados {processed_count}/{total_files_to_process} arquivos...")
                except FileNotFoundError:
                    logger.error(f"  erro: arquivo de origem não encontrado para {image_index} em {source_path}. pulando.")
                except Exception as e:
                    logger.error(f"  erro ao processar {image_index} de {source_path} para {dest_path}: {e}")
        
        logger.info(f"processamento e cópia de imagens concluídos. total de arquivos processados: {processed_count}")
        logger.info("processamento de imagem finalizado.")

    def process_images_pipeline(self):
        """orquestra todo o pipeline de processamento de imagens."""
        processed_df = self.load_processed_data()
        if processed_df is None:
            logger.error("processamento de imagem abortado devido a falha no carregamento dos dados processados.")
            return

        image_path_map = self.build_image_path_map()
        processed_df = self.add_source_path_to_df(processed_df, image_path_map)

        if processed_df.empty:
            logger.error("nenhuma imagem para processar após filtrar por caminhos de origem ausentes. abortando processamento de imagem.")
            return

        train_df, val_df, test_df = self.split_data(processed_df)
        self.create_destination_directories()
        self.resize_and_copy_images(train_df, val_df, test_df)

if __name__ == "__main__":
    logger.info("executando exemplo de imageprocessor...")
    
    test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
    os.makedirs(test_data_dir, exist_ok=True)
    test_processed_data_filepath = os.path.join(test_data_dir, PROCESSED_DATA_FILENAME)

    if not os.path.exists(test_processed_data_filepath):
        logger.info(f"criando um '{PROCESSED_DATA_FILENAME}' dummy para testar o imageprocessor...")
        dummy_processed_data = {
            IMAGE_INDEX_COL: [f'000000{i:02d}_000.png' for i in range(1, 101)],
            FINDING_LABELS_COL: [i % 2 for i in range(100)],
            VIEW_POSITION_COL: [i % 2 for i in range(100)],
            STRATIFY_COL: [f'{i % 2}_{i % 2}' for i in range(100)]
        }
        pd.DataFrame(dummy_processed_data).to_csv(test_processed_data_filepath, index=False)
        logger.info("arquivo de dados processados dummy criado.")

    processor = ImageProcessor(data_path=test_data_dir)
    processor.process_images_pipeline()