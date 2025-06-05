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
    """avalia um modelo keras treinado e gera métricas de desempenho (relatório de classificação e matriz de confusão)."""
    def __init__(self, models_dir=MODELS_DIR, reports_dir=REPORTS_DIR, checkpoint_filename=MODEL_CHECKPOINT_FILENAME, target_labels=TARGET_LABELS):
        self.checkpoint_filepath = os.path.join(models_dir, checkpoint_filename)
        self.reports_dir = reports_dir
        self.target_labels = target_labels
        self.best_model = None
        
        os.makedirs(self.reports_dir, exist_ok=True)
        logger.info(f"modelevaluator inicializado. checkpoint do modelo esperado em: {self.checkpoint_filepath}")
        logger.info(f"relatórios de avaliação serão salvos em: {self.reports_dir}")


    def load_best_model(self):
        """carrega o melhor modelo treinado do arquivo de checkpoint."""
        logger.info(f"carregando o melhor modelo de '{self.checkpoint_filepath}'...")
        if not os.path.exists(self.checkpoint_filepath):
            logger.error(f"erro: arquivo de checkpoint do modelo não encontrado em '{self.checkpoint_filepath}'.")
            logger.error("certifique-se de que o modelo foi treinado e salvo.")
            self.best_model = None
            return False
        try:
            self.best_model = keras.models.load_model(self.checkpoint_filepath)
            logger.info("melhor modelo carregado com sucesso.")
            return True
        except Exception as e:
            logger.error(f"erro ao carregar o modelo de '{self.checkpoint_filepath}': {e}")
            self.best_model = None
            return False

    def evaluate_model(self, test_ds: tf.data.Dataset):
        """avalia o modelo carregado no conjunto de dados de teste."""
        if self.best_model is None:
            logger.error("nenhum modelo carregado para avaliação. por favor, carregue o modelo primeiro.")
            return
        
        logger.info("\navaliando o modelo no conjunto de teste...")
        try:
            loss, accuracy = self.best_model.evaluate(test_ds)
            logger.info(f"perda do teste: {loss:.4f}")
            logger.info(f"acurácia do teste: {accuracy:.4f}")
        except Exception as e:
            logger.error(f"erro durante a avaliação do modelo: {e}")

    def generate_classification_report(self, test_ds: tf.data.Dataset):
        """gera e imprime o relatório de classificação e o salva em um arquivo."""
        if self.best_model is None:
            logger.error("nenhum modelo carregado para relatório de classificação. por favor, carregue o modelo primeiro.")
            return

        logger.info("\ngerando relatório de classificação...")
        try:
            test_labels = np.concatenate([y for x, y in test_ds], axis=0)
            test_predictions = self.best_model.predict(test_ds)
            test_predictions_binary = (test_predictions > 0.5).astype(int)

            report = classification_report(test_labels, test_predictions_binary,
                                           target_names=[f'classe {name}' for name in self.target_labels])
            logger.info("\nrelatório de classificação:\n" + report)

            report_filepath = os.path.join(self.reports_dir, EVALUATION_CLASSIFICATION_REPORT_FILENAME)
            with open(report_filepath, 'w') as f:
                f.write(f"a perda do teste e a acurácia do teste de model.evaluate() podem diferir ligeiramente do relatório devido a agrupamento/média.\n")
                f.write("considere registrar esses valores aqui se necessário.\n\n")
                f.write(report)
            logger.info(f"relatório de classificação salvo em '{report_filepath}'")

        except Exception as e:
            logger.error(f"erro ao gerar ou salvar relatório de classificação: {e}")

    def plot_confusion_matrix(self, test_ds: tf.data.Dataset):
        """gera, plota e salva a matriz de confusão."""
        if self.best_model is None:
            logger.error("nenhum modelo carregado para matriz de confusão. por favor, carregue o modelo primeiro.")
            return

        logger.info("\ngerando matriz de confusão...")
        try:
            test_labels = np.concatenate([y for x, y in test_ds], axis=0)
            test_predictions = self.best_model.predict(test_ds)
            test_predictions_binary = (test_predictions > 0.5).astype(int)

            cm = confusion_matrix(test_labels, test_predictions_binary)
            logger.info("\nmatriz de confusão:\n" + str(cm))

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=[f'{self.target_labels[0]} (sem efusão)', f'{self.target_labels[1]} (efusão)'],
                        yticklabels=[f'{self.target_labels[0]} (sem efusão)', f'{self.target_labels[1]} (efusão)'])
            plt.xlabel('rótulo previsto')
            plt.ylabel('rótulo verdadeiro')
            plt.title('matriz de confusão')
            
            plot_filepath = os.path.join(self.reports_dir, EVALUATION_CONFUSION_MATRIX_FILENAME)
            plt.savefig(plot_filepath)
            plt.close()
            logger.info(f"gráfico da matriz de confusão salvo em '{plot_filepath}'")

        except Exception as e:
            logger.error(f"erro ao plotar ou salvar matriz de confusão: {e}")

    def run_evaluation(self, test_ds: tf.data.Dataset):
        """orquestra todo o pipeline de avaliação do modelo."""
        if self.load_best_model():
            self.evaluate_model(test_ds)
            self.generate_classification_report(test_ds)
            self.plot_confusion_matrix(test_ds)
        else:
            logger.error("avaliação do modelo abortada devido a falha no carregamento do melhor modelo.")

if __name__ == "__main__":
    logger.info("executando exemplo de modelevaluator...")
    from src.models.model_builder import ModelBuilder
    from src.utils.constants import IMAGE_SIZE, BATCH_SIZE

    builder = ModelBuilder()
    dummy_model = builder.build_transfer_learning_model()
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    dummy_model_path = os.path.join(MODELS_DIR, MODEL_CHECKPOINT_FILENAME)
    dummy_model.save(dummy_model_path)
    logger.info(f"modelo dummy salvo em {dummy_model_path} para testar o avaliador.")

    # 2. cria um conjunto de dados de teste dummy
    def create_dummy_dataset(num_samples, image_size_tuple, batch_size_val):
        images = tf.random.normal((num_samples, image_size_tuple[0], image_size_tuple[1], 3))
        labels = tf.cast(tf.random.uniform((num_samples,), minval=0, maxval=2, dtype=tf.int32), tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.batch(batch_size_val).cache().prefetch(tf.data.AUTOTUNE)
        return dataset

    dummy_test_ds = create_dummy_dataset(50, IMAGE_SIZE, BATCH_SIZE)

    # 3. inicializa e executa o avaliador
    evaluator = ModelEvaluator()
    evaluator.run_evaluation(dummy_test_ds)

    if os.path.exists(dummy_model_path):
         os.remove(dummy_model_path)