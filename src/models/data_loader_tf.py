import tensorflow as tf
import os
from src.utils.logger import logger
from src.utils.constants import DATA_DIR, IMAGE_SIZE, BATCH_SIZE, VIEW_POSITION_FOR_MODEL, TARGET_LABELS

class TFDataLoader:
    """carrega datasets de imagens usando image_dataset_from_directory do tensorflow."""
    def __init__(self, data_path=DATA_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, 
                 view_position=VIEW_POSITION_FOR_MODEL, target_labels=TARGET_LABELS):
        self.data_path = data_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.view_position = view_position
        self.target_labels = target_labels
        logger.info(f"tfdataloader inicializado para posição de visualização: {self.view_position}")

    def _check_directories(self, *dirs):
        """verifica se todos os diretórios especificados existem."""
        for d in dirs:
            if not os.path.exists(d):
                logger.error(f"erro: diretório '{d}' não encontrado.")
                logger.error("certifique-se de que o processamento de imagem foi concluído para criar esses diretórios.")
                return False
        return True

    def load_datasets(self):
        """carrega os conjuntos de dados de treinamento, validação e teste dos diretórios estruturados."""
        train_dir = os.path.join(self.data_path, 'train', self.view_position)
        validation_dir = os.path.join(self.data_path, 'validation', self.view_position)
        test_dir = os.path.join(self.data_path, 'test', self.view_position)

        logger.info(f"carregando datasets para a posição de visualização '{self.view_position}' de:")
        logger.info(f"  treino: {train_dir}")
        logger.info(f"  validação: {validation_dir}")
        logger.info(f"  teste: {test_dir}")

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
            logger.info(f"dataset de treino carregado com {len(train_ds) * self.batch_size} imagens (aprox).")

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
            logger.info(f"dataset de validação carregado com {len(validation_ds) * self.batch_size} imagens (aprox).")

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
            logger.info(f"dataset de teste carregado com {len(test_ds) * self.batch_size} imagens (aprox).")

            # pré-busca de dados para melhor desempenho
            autotune = tf.data.AUTOTUNE
            train_ds = train_ds.cache().prefetch(buffer_size=autotune)
            validation_ds = validation_ds.cache().prefetch(buffer_size=autotune)
            test_ds = test_ds.cache().prefetch(buffer_size=autotune)
            logger.info("datasets em cache e pré-buscados.")

            return train_ds, validation_ds, test_ds

        except Exception as e:
            logger.error(f"ocorreu um erro durante o carregamento do dataset: {e}")
            return None, None, None

if __name__ == "__main__":
    logger.info("executando exemplo de tfdataloader...")
    test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
    
    dummy_train_dir = os.path.join(test_data_dir, 'train', VIEW_POSITION_FOR_MODEL, TARGET_LABELS[0])
    dummy_val_dir = os.path.join(test_data_dir, 'validation', VIEW_POSITION_FOR_MODEL, TARGET_LABELS[0])
    dummy_test_dir = os.path.join(test_data_dir, 'test', VIEW_POSITION_FOR_MODEL, TARGET_LABELS[0])

    os.makedirs(dummy_train_dir, exist_ok=True)
    os.makedirs(dummy_val_dir, exist_ok=True)
    os.makedirs(dummy_test_dir, exist_ok=True)

    # cria arquivos de imagem dummy
    from PIL import Image, ImageDraw
    for i in range(5):
        img = Image.new('RGB', IMAGE_SIZE, color = (i*10, i*20, i*30))
        d = ImageDraw.Draw(img)
        d.text((10,10), f"dummy {i}", fill=(255,255,0))
        img.save(os.path.join(dummy_train_dir, f'dummy_train_{i}.png'))
        img.save(os.path.join(dummy_val_dir, f'dummy_val_{i}.png'))
        img.save(os.path.join(dummy_test_dir, f'dummy_test_{i}.png'))
    
    # também cria para o segundo rótulo (target_labels[1])
    dummy_train_dir_1 = os.path.join(test_data_dir, 'train', VIEW_POSITION_FOR_MODEL, TARGET_LABELS[1])
    dummy_val_dir_1 = os.path.join(test_data_dir, 'validation', VIEW_POSITION_FOR_MODEL, TARGET_LABELS[1])
    dummy_test_dir_1 = os.path.join(test_data_dir, 'test', VIEW_POSITION_FOR_MODEL, TARGET_LABELS[1])

    os.makedirs(dummy_train_dir_1, exist_ok=True)
    os.makedirs(dummy_val_dir_1, exist_ok=True)
    os.makedirs(dummy_test_dir_1, exist_ok=True)

    for i in range(5):
        img = Image.new('RGB', IMAGE_SIZE, color = (255 - i*10, 255 - i*20, 255 - i*30))
        d = ImageDraw.Draw(img)
        d.text((10,10), f"dummy {i}", fill=(0,0,255))
        img.save(os.path.join(dummy_train_dir_1, f'dummy_train_label1_{i}.png'))
        img.save(os.path.join(dummy_val_dir_1, f'dummy_val_label1_{i}.png'))
        img.save(os.path.join(dummy_test_dir_1, f'dummy_test_label1_{i}.png'))

    loader = TFDataLoader(data_path=test_data_dir)
    train_ds, val_ds, test_ds = loader.load_datasets()

    if train_ds and val_ds and test_ds:
        logger.info("todos os datasets carregados com sucesso.")
        for images, labels in train_ds.take(1):
            logger.info(f"formato de um lote de imagens: {images.shape}")
            logger.info(f"formato de um lote de rótulos: {labels.shape}")
    else:
        logger.error("falha ao carregar datasets.")

    # Limpa diretórios dummy (opcional)
    # import shutil
    # shutil.rmtree(os.path.join(test_data_dir, 'train'))
    # shutil.rmtree(os.path.join(test_data_dir, 'validation'))
    # shutil.rmtree(os.path.join(test_data_dir, 'test'))