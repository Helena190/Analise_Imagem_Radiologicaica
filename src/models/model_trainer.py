import tensorflow as tf
from tensorflow import keras
import os
from src.utils.logger import logger
from src.utils.constants import EPOCHS, MODELS_DIR, MODEL_CHECKPOINT_FILENAME

class ModelTrainer:
    """
    Gerencia o processo de treinamento de um modelo Keras.
    """
    def __init__(self, epochs=EPOCHS, models_dir=MODELS_DIR, checkpoint_filename=MODEL_CHECKPOINT_FILENAME):
        self.epochs = epochs
        self.checkpoint_filepath = os.path.join(models_dir, checkpoint_filename)
        os.makedirs(models_dir, exist_ok=True)
        logger.info(f"ModelTrainer initialized. Model checkpoints will be saved to: {self.checkpoint_filepath}")

    def train_model(self, model: keras.Model, train_ds: tf.data.Dataset, validation_ds: tf.data.Dataset):
        """
        Treina o modelo Keras fornecido usando os conjuntos de dados dados.

        Args:
            model (keras.Model): O modelo Keras compilado para treinar.
            train_ds (tf.data.Dataset): O conjunto de dados de treinamento.
            validation_ds (tf.data.Dataset): O conjunto de dados de validação.

        Returns:
            keras.callbacks.History: O objeto de histórico do treinamento do modelo.
        """
        logger.info("Starting model training...")

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_filepath, save_best_only=True, monitor='val_loss')

        try:
            history = model.fit(
                train_ds,
                epochs=self.epochs,
                validation_data=validation_ds,
                callbacks=[early_stopping, model_checkpoint]
            )
            logger.info("Model training finished.")
            return history
        except Exception as e:
            logger.error(f"An error occurred during model training: {e}")
            return None
