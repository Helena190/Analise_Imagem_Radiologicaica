import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from src.utils.logger import logger
from src.utils.constants import MODELS_DIR, MODEL_CHECKPOINT_FILENAME, TARGET_LABELS

class ModelEvaluator:
    """
    Avalia um modelo Keras treinado e gera métricas de desempenho
    como relatório de classificação e matriz de confusão.
    """
    def __init__(self, models_dir=MODELS_DIR, checkpoint_filename=MODEL_CHECKPOINT_FILENAME, target_labels=TARGET_LABELS):
        self.checkpoint_filepath = os.path.join(models_dir, checkpoint_filename)
        self.target_labels = target_labels
        self.best_model = None
        logger.info(f"ModelEvaluator initialized. Model checkpoint expected at: {self.checkpoint_filepath}")

    def load_best_model(self):
        """
        Carrega o melhor modelo treinado do arquivo de checkpoint.
        """
        logger.info(f"Loading best model from '{self.checkpoint_filepath}'...")
        if not os.path.exists(self.checkpoint_filepath):
            logger.error(f"Error: Model checkpoint file not found at '{self.checkpoint_filepath}'.")
            logger.error("Please ensure the model has been trained and saved.")
            self.best_model = None
            return False
        try:
            self.best_model = keras.models.load_model(self.checkpoint_filepath)
            logger.info("Best model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Error loading model from '{self.checkpoint_filepath}': {e}")
            self.best_model = None
            return False

    def evaluate_model(self, test_ds: tf.data.Dataset):
        """
        Avalia o modelo carregado no conjunto de dados de teste.
        """
        if self.best_model is None:
            logger.error("No model loaded for evaluation. Please load the model first.")
            return

        logger.info("\nEvaluating the model on the test set...")
        try:
            loss, accuracy = self.best_model.evaluate(test_ds)
            logger.info(f"Test Loss: {loss:.4f}")
            logger.info(f"Test Accuracy: {accuracy:.4f}")
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")

    def generate_classification_report(self, test_ds: tf.data.Dataset):
        """
        Gera e imprime o relatório de classificação.
        """
        if self.best_model is None:
            logger.error("No model loaded for classification report. Please load the model first.")
            return

        logger.info("\nGenerating classification report...")
        try:
            test_labels = np.concatenate([y for x, y in test_ds], axis=0)
            test_predictions = self.best_model.predict(test_ds)
            test_predictions_binary = (test_predictions > 0.5).astype(int)

            report = classification_report(test_labels, test_predictions_binary, 
                                           target_names=[f'Class {name}' for name in self.target_labels])
            logger.info("\nClassification Report:\n" + report)
        except Exception as e:
            logger.error(f"Error generating classification report: {e}")

    def plot_confusion_matrix(self, test_ds: tf.data.Dataset):
        """
        Gera e plota a matriz de confusão.
        """
        if self.best_model is None:
            logger.error("No model loaded for confusion matrix. Please load the model first.")
            return

        logger.info("\nGenerating confusion matrix...")
        try:
            test_labels = np.concatenate([y for x, y in test_ds], axis=0)
            test_predictions = self.best_model.predict(test_ds)
            test_predictions_binary = (test_predictions > 0.5).astype(int)

            cm = confusion_matrix(test_labels, test_predictions_binary)
            logger.info("\nConfusion Matrix:\n" + str(cm))

            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=[f'Class {name}' for name in self.target_labels], 
                        yticklabels=[f'Class {name}' for name in self.target_labels])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")

    def run_evaluation(self, test_ds: tf.data.Dataset):
        """
        Orquestra todo o pipeline de avaliação do modelo.
        """
        if self.load_best_model():
            self.evaluate_model(test_ds)
            self.generate_classification_report(test_ds)
            self.plot_confusion_matrix(test_ds)
        else:
            logger.error("Model evaluation aborted due to failure in loading the best model.")

if __name__ == "__main__":
    logger.info("Running ModelEvaluator example...")
    # Este exemplo requer um modelo e conjuntos de dados dummy.
    # Em um cenário real, você os obteria de ModelBuilder e TFDataLoader.

    # 1. Cria um modelo dummy e o salva (imitando um modelo treinado)
    from src.models.model_builder import ModelBuilder
    builder = ModelBuilder()
    dummy_model = builder.build_transfer_learning_model()
    
    # Garante que o diretório de modelos exista para salvar o modelo dummy
    os.makedirs(MODELS_DIR, exist_ok=True)
    dummy_model_path = os.path.join(MODELS_DIR, MODEL_CHECKPOINT_FILENAME)
    dummy_model.save(dummy_model_path)
    logger.info(f"Dummy model saved to {dummy_model_path} for testing evaluator.")

    # 2. Cria um conjunto de dados de teste dummy (simplificado para exemplo)
    def create_dummy_dataset(num_samples, image_size, batch_size):
        images = tf.random.normal((num_samples, image_size[0], image_size[1], 3))
        # Cria rótulos que permitem uma mistura de 0s e 1s para uma matriz de confusão significativa
        labels = tf.cast(tf.random.uniform((num_samples,), minval=0, maxval=2, dtype=tf.int32), tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
        return dataset

    dummy_test_ds = create_dummy_dataset(50, IMAGE_SIZE, BATCH_SIZE)

    # 3. Inicializa e executa o avaliador
    evaluator = ModelEvaluator()
    evaluator.run_evaluation(dummy_test_ds)

    # Limpa o arquivo do modelo dummy (opcional)
    # os.remove(dummy_model_path)