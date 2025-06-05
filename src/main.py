# src/main.py
import logging
import argparse
import os # Adicionado
from src.utils.logger import setup_logging, logger
from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.data.data_saver import DataSaver
from src.data.image_processor import ImageProcessor
from src.analysis.descriptive_statistics import DescriptiveStatistics
from src.models.data_loader_tf import TFDataLoader
from src.models.model_builder import ModelBuilder
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.utils.constants import VIEW_POSITION_FOR_MODEL, REPORTS_DIR # Adicionado REPORTS_DIR

def run_data_pipeline():
    """Executa o pipeline de extração de dados e processamento de imagens."""
    logger.info("--- Starting Data Pipeline ---")
    
    data_loader = DataLoader()
    original_df = data_loader.load_original_data()

    if original_df is None:
        logger.error("Failed to load original data. Aborting data pipeline.")
        return

    data_processor = DataProcessor()
    processed_df = data_processor.process_data(original_df.copy()) 

    data_saver = DataSaver()
    data_saver.save_processed_data(processed_df)
    data_saver.save_source_data_dictionary()
    data_saver.save_processed_data_dictionary(processed_df)

    image_processor = ImageProcessor()
    image_processor.process_images_pipeline()
    
    logger.info("--- Data Pipeline Completed ---")

def run_analysis_pipeline():
    """Executa o pipeline de análise de estatísticas descritivas."""
    logger.info("--- Starting Analysis Pipeline ---")
    stats_analyzer = DescriptiveStatistics()
    stats_analyzer.run_analysis()
    logger.info("--- Analysis Pipeline Completed ---")

def run_model_pipeline():
    """Executa o pipeline de treinamento e avaliação do modelo de aprendizado de máquina."""
    logger.info("--- Starting Model Pipeline ---")
    
    tf_data_loader = TFDataLoader(view_position=VIEW_POSITION_FOR_MODEL)
    train_ds, validation_ds, test_ds = tf_data_loader.load_datasets()

    if train_ds is None or validation_ds is None or test_ds is None:
        logger.error("Failed to load TensorFlow datasets. Aborting model pipeline.")
        return

    model_builder = ModelBuilder()
    model = model_builder.build_transfer_learning_model()

    model_trainer = ModelTrainer()
    history = model_trainer.train_model(model, train_ds, validation_ds)

    if history is None:
        logger.error("Model training failed. Aborting model evaluation.")
        return

    # A avaliação aqui já salva os relatórios devido às modificações no ModelEvaluator
    model_evaluator = ModelEvaluator() 
    model_evaluator.run_evaluation(test_ds)

    logger.info("--- Model Pipeline Completed ---")

def run_evaluation_only_pipeline(): # Nova função
    """Executa apenas a avaliação de um modelo treinado existente e salva os relatórios."""
    logger.info("--- Starting Evaluation-Only Pipeline ---")
    
    # 1. Carrega os dados de teste
    tf_data_loader = TFDataLoader(view_position=VIEW_POSITION_FOR_MODEL)
    # Precisamos apenas do conjunto de teste para avaliação
    # _ ignora train_ds e validation_ds
    _, _, test_ds = tf_data_loader.load_datasets() 

    if test_ds is None:
        logger.error("Failed to load TensorFlow test dataset. Aborting evaluation-only pipeline.")
        return

    # 2. Inicializa o ModelEvaluator
    # O ModelEvaluator irá carregar o modelo treinado e salvar os relatórios no diretório correto
    # Garante que o diretório de relatórios exista (ModelEvaluator também faz isso, mas não custa)
    os.makedirs(REPORTS_DIR, exist_ok=True) 
    model_evaluator = ModelEvaluator()

    # 3. Executa a avaliação
    # Este método agora carrega o modelo, avalia e salva os relatórios (matriz e relatório de class.)
    model_evaluator.run_evaluation(test_ds)
    
    logger.info("--- Evaluation-Only Pipeline Completed ---")
    logger.info(f"Evaluation reports (e.g., confusion matrix, classification report) saved to '{REPORTS_DIR}' directory.")

def main():
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline MVP")
    parser.add_argument('--data', action='store_true', help='Run the data extraction and image processing pipeline.')
    parser.add_argument('--analyze', action='store_true', help='Run the descriptive statistics analysis pipeline.')
    parser.add_argument('--model', action='store_true', help='Run the machine learning model training and evaluation pipeline.')
    parser.add_argument('--evaluate_only', action='store_true', help='Run only model evaluation on the test set and save reports (does not retrain).') # Novo argumento
    parser.add_argument('--all', action='store_true', help='Run all pipelines (data, analysis, model).')
    parser.add_argument('--log_level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level.')

    args = parser.parse_args()

    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    logger.setLevel(numeric_level)
    logger.info(f"Logging level set to {args.log_level}")

    if args.all:
        run_data_pipeline()
        run_analysis_pipeline()
        run_model_pipeline() # O pipeline do modelo já treina e avalia (salvando relatórios)
    elif args.data:
        run_data_pipeline()
    elif args.analyze:
        run_analysis_pipeline()
    elif args.model:
        run_model_pipeline() # O pipeline do modelo já treina e avalia (salvando relatórios)
    elif args.evaluate_only: # Nova condição
        run_evaluation_only_pipeline()
    else:
        parser.print_help()
        logger.info("No pipeline specified. Use --data, --analyze, --model, --evaluate_only, or --all.")

if __name__ == "__main__":
    setup_logging() 
    main()