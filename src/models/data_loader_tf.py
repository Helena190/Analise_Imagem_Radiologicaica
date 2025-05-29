import tensorflow as tf
import os
from src.utils.logger import logger
from src.utils.constants import DATA_DIR, IMAGE_SIZE, BATCH_SIZE, VIEW_POSITION_FOR_MODEL, TARGET_LABELS

class TFDataLoader:
    """
    Lida com o carregamento de conjuntos de dados de imagens usando image_dataset_from_directory do TensorFlow.
    """
    def __init__(self, data_path=DATA_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, 
                 view_position=VIEW_POSITION_FOR_MODEL, target_labels=TARGET_LABELS):
        self.data_path = data_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.view_position = view_position
        self.target_labels = target_labels
        logger.info(f"TFDataLoader initialized for view position: {self.view_position}")

    def _check_directories(self, *dirs):
        """Função auxiliar para verificar se todos os diretórios especificados existem."""
        for d in dirs:
            if not os.path.exists(d):
                logger.error(f"Error: Directory '{d}' not found.")
                logger.error("Please ensure image processing has been completed to create these directories.")
                return False
        return True

    def load_datasets(self):
        """
        Carrega os conjuntos de dados de treinamento, validação e teste dos diretórios estruturados.

        Retorna:
            tuple: (train_ds, validation_ds, test_ds) ou (None, None, None) se os diretórios estiverem faltando.
        """
        train_dir = os.path.join(self.data_path, 'train', self.view_position)
        validation_dir = os.path.join(self.data_path, 'validation', self.view_position)
        test_dir = os.path.join(self.data_path, 'test', self.view_position)

        logger.info(f"Loading datasets for view position '{self.view_position}' from:")
        logger.info(f"  Train: {train_dir}")
        logger.info(f"  Validation: {validation_dir}")
        logger.info(f"  Test: {test_dir}")

        if not self._check_directories(train_dir, validation_dir, test_dir):
            return None, None, None

        try:
            train_ds = tf.keras.utils.image_dataset_from_directory(
                train_dir,
                labels='inferred',
                label_mode='binary',
                image_size=self.image_size,
                interpolation='nearest',
                batch_size=self.batch_size,
                shuffle=True,
                seed=42,
                class_names=self.target_labels
            )
            logger.info(f"Loaded train dataset with {len(train_ds) * self.batch_size} images (approx).")

            validation_ds = tf.keras.utils.image_dataset_from_directory(
                validation_dir,
                labels='inferred',
                label_mode='binary',
                image_size=self.image_size,
                interpolation='nearest',
                batch_size=self.batch_size,
                shuffle=False,
                seed=42,
                class_names=self.target_labels
            )
            logger.info(f"Loaded validation dataset with {len(validation_ds) * self.batch_size} images (approx).")

            test_ds = tf.keras.utils.image_dataset_from_directory(
                test_dir,
                labels='inferred',
                label_mode='binary',
                image_size=self.image_size,
                interpolation='nearest',
                batch_size=self.batch_size,
                shuffle=False,
                seed=42,
                class_names=self.target_labels
            )
            logger.info(f"Loaded test dataset with {len(test_ds) * self.batch_size} images (approx).")

            # Pré-busca de dados para melhor desempenho
            AUTOTUNE = tf.data.AUTOTUNE
            train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
            validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)
            test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
            logger.info("Datasets cached and prefetched.")

            return train_ds, validation_ds, test_ds

        except Exception as e:
            logger.error(f"An error occurred during dataset loading: {e}")
            return None, None, None

if __name__ == "__main__":
    logger.info("Running TFDataLoader example...")
    # Para um teste completo, você precisaria dos diretórios de imagem criados pelo ImageProcessor
    # Cria diretórios e arquivos dummy para teste
    test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
    
    # Define caminhos para dados dummy
    dummy_train_dir = os.path.join(test_data_dir, 'train', VIEW_POSITION_FOR_MODEL, TARGET_LABELS[0])
    dummy_val_dir = os.path.join(test_data_dir, 'validation', VIEW_POSITION_FOR_MODEL, TARGET_LABELS[0])
    dummy_test_dir = os.path.join(test_data_dir, 'test', VIEW_POSITION_FOR_MODEL, TARGET_LABELS[0])

    os.makedirs(dummy_train_dir, exist_ok=True)
    os.makedirs(dummy_val_dir, exist_ok=True)
    os.makedirs(dummy_test_dir, exist_ok=True)

    # Cria arquivos de imagem dummy
    from PIL import Image, ImageDraw
    for i in range(5): # Create 5 dummy images for each split/label
        img = Image.new('RGB', IMAGE_SIZE, color = (i*10, i*20, i*30))
        d = ImageDraw.Draw(img)
        d.text((10,10), f"Dummy {i}", fill=(255,255,0))
        img.save(os.path.join(dummy_train_dir, f'dummy_train_{i}.png'))
        img.save(os.path.join(dummy_val_dir, f'dummy_val_{i}.png'))
        img.save(os.path.join(dummy_test_dir, f'dummy_test_{i}.png'))
    
    # Também cria para o segundo rótulo (TARGET_LABELS[1])
    dummy_train_dir_1 = os.path.join(test_data_dir, 'train', VIEW_POSITION_FOR_MODEL, TARGET_LABELS[1])
    dummy_val_dir_1 = os.path.join(test_data_dir, 'validation', VIEW_POSITION_FOR_MODEL, TARGET_LABELS[1])
    dummy_test_dir_1 = os.path.join(test_data_dir, 'test', VIEW_POSITION_FOR_MODEL, TARGET_LABELS[1])

    os.makedirs(dummy_train_dir_1, exist_ok=True)
    os.makedirs(dummy_val_dir_1, exist_ok=True)
    os.makedirs(dummy_test_dir_1, exist_ok=True)

    for i in range(5):
        img = Image.new('RGB', IMAGE_SIZE, color = (255 - i*10, 255 - i*20, 255 - i*30))
        d = ImageDraw.Draw(img)
        d.text((10,10), f"Dummy {i}", fill=(0,0,255))
        img.save(os.path.join(dummy_train_dir_1, f'dummy_train_label1_{i}.png'))
        img.save(os.path.join(dummy_val_dir_1, f'dummy_val_label1_{i}.png'))
        img.save(os.path.join(dummy_test_dir_1, f'dummy_test_label1_{i}.png'))

    loader = TFDataLoader(data_path=test_data_dir)
    train_ds, val_ds, test_ds = loader.load_datasets()

    if train_ds and val_ds and test_ds:
        logger.info("Successfully loaded all datasets.")
        # Você pode iterar e verificar um lote
        for images, labels in train_ds.take(1):
            logger.info(f"Shape of a batch of images: {images.shape}")
            logger.info(f"Shape of a batch of labels: {labels.shape}")
    else:
        logger.error("Failed to load datasets.")

    # Limpa diretórios dummy (opcional)
    # import shutil
    # shutil.rmtree(os.path.join(test_data_dir, 'train'))
    # shutil.rmtree(os.path.join(test_data_dir, 'validation'))
    # shutil.rmtree(os.path.join(test_data_dir, 'test'))