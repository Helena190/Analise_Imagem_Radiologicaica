# src/models/model_evaluator.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from src.utils.logger import logger
from src.utils.constants import (
    MODELS_DIR, 
    MODEL_CHECKPOINT_FILENAME, 
    TARGET_LABELS,
    REPORTS_DIR, # Adicionado
    EVALUATION_CONFUSION_MATRIX_FILENAME, # Adicionado
    EVALUATION_CLASSIFICATION_REPORT_FILENAME # Adicionado
)

class ModelEvaluator:
    """
    Avalia um modelo Keras treinado e gera métricas de desempenho
    como relatório de classificação e matriz de confusão.
    Salva os relatórios gerados.
    """
    def __init__(self, models_dir=MODELS_DIR, reports_dir=REPORTS_DIR, checkpoint_filename=MODEL_CHECKPOINT_FILENAME, target_labels=TARGET_LABELS):
        self.checkpoint_filepath = os.path.join(models_dir, checkpoint_filename)
        self.reports_dir = reports_dir # Adicionado
        self.target_labels = target_labels
        self.best_model = None
        
        # Garante que o diretório de relatórios exista
        os.makedirs(self.reports_dir, exist_ok=True) # Adicionado
        logger.info(f"ModelEvaluator initialized. Model checkpoint expected at: {self.checkpoint_filepath}")
        logger.info(f"Evaluation reports will be saved to: {self.reports_dir}")


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
        Gera e imprime o relatório de classificação e o salva em um arquivo.
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

            # Salva o relatório em arquivo
            report_filepath = os.path.join(self.reports_dir, EVALUATION_CLASSIFICATION_REPORT_FILENAME)
            with open(report_filepath, 'w') as f:
                f.write(f"Test Loss, Test Accuracy from model.evaluate() might differ slightly from report due to batching/averaging.\n")
                f.write("Consider logging those values here if needed.\n\n")
                f.write(report)
            logger.info(f"Classification report saved to '{report_filepath}'")

        except Exception as e:
            logger.error(f"Error generating or saving classification report: {e}")

    def plot_confusion_matrix(self, test_ds: tf.data.Dataset):
        """
        Gera, plota e salva a matriz de confusão.
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

            plt.figure(figsize=(8, 6)) # Aumentado um pouco para melhor visualização
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=[f'{self.target_labels[0]} (No Efus.)', f'{self.target_labels[1]} (Efus.)'], 
                        yticklabels=[f'{self.target_labels[0]} (No Efus.)', f'{self.target_labels[1]} (Efus.)'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            
            # Salva a figura
            plot_filepath = os.path.join(self.reports_dir, EVALUATION_CONFUSION_MATRIX_FILENAME)
            plt.savefig(plot_filepath)
            plt.close() # Fecha a figura para liberar memória
            logger.info(f"Confusion matrix plot saved to '{plot_filepath}'")

        except Exception as e:
            logger.error(f"Error plotting or saving confusion matrix: {e}")

    def run_evaluation(self, test_ds: tf.data.Dataset):
        """
        Orquestra todo o pipeline de avaliação do modelo.
        """
        if self.load_best_model():
            self.evaluate_model(test_ds) # Loga perda e acurácia
            self.generate_classification_report(test_ds) # Gera e salva relatório
            self.plot_confusion_matrix(test_ds) # Gera e salva matriz
        else:
            logger.error("Model evaluation aborted due to failure in loading the best model.")

if __name__ == "__main__":
    logger.info("Running ModelEvaluator example...")
    # Este exemplo requer um modelo e conjuntos de dados dummy.
    # Em um cenário real, você os obteria de ModelBuilder e TFDataLoader.

    # 1. Cria um modelo dummy e o salva (imitando um modelo treinado)
    from src.models.model_builder import ModelBuilder # Mova para o topo se for usar globalmente
    from src.utils.constants import IMAGE_SIZE, BATCH_SIZE # Mova para o topo

    builder = ModelBuilder()
    dummy_model = builder.build_transfer_learning_model()
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    dummy_model_path = os.path.join(MODELS_DIR, MODEL_CHECKPOINT_FILENAME)
    dummy_model.save(dummy_model_path)
    logger.info(f"Dummy model saved to {dummy_model_path} for testing evaluator.")

    # 2. Cria um conjunto de dados de teste dummy (simplificado para exemplo)
    def create_dummy_dataset(num_samples, image_size_tuple, batch_size_val):
        images = tf.random.normal((num_samples, image_size_tuple[0], image_size_tuple[1], 3))
        labels = tf.cast(tf.random.uniform((num_samples,), minval=0, maxval=2, dtype=tf.int32), tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.batch(batch_size_val).cache().prefetch(tf.data.AUTOTUNE)
        return dataset

    dummy_test_ds = create_dummy_dataset(50, IMAGE_SIZE, BATCH_SIZE)

    # 3. Inicializa e executa o avaliador
    evaluator = ModelEvaluator() # reports_dir será o padrão de constants.py
    evaluator.run_evaluation(dummy_test_ds)

    # Limpa o arquivo do modelo dummy (opcional)
    if os.path.exists(dummy_model_path):
         os.remove(dummy_model_path)