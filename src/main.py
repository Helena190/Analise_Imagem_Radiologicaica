import logging
import argparse
import os
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
from src.utils.constants import VIEW_POSITION_FOR_MODEL, REPORTS_DIR

def run_data_pipeline():
    """executa o pipeline de dados."""
    logger.info("--- iniciando pipeline de dados ---")
    
    data_loader = DataLoader()
    original_df = data_loader.load_original_data()

    if original_df is None:
        logger.error("falha ao carregar dados originais. abortando pipeline de dados.")
        return

    data_processor = DataProcessor()
    processed_df = data_processor.process_data(original_df.copy()) 

    data_saver = DataSaver()
    data_saver.save_processed_data(processed_df)
    data_saver.save_source_data_dictionary()
    data_saver.save_processed_data_dictionary(processed_df)

    image_processor = ImageProcessor()
    image_processor.process_images_pipeline()
    
    logger.info("--- pipeline de dados concluído ---")

def run_analysis_pipeline():
    """executa o pipeline de análise."""
    logger.info("--- iniciando pipeline de análise ---")
    stats_analyzer = DescriptiveStatistics()
    stats_analyzer.run_analysis()
    logger.info("--- pipeline de análise concluído ---")

def run_model_pipeline():
    """executa o pipeline do modelo."""
    logger.info("--- iniciando pipeline de modelo ---")
    
    tf_data_loader = TFDataLoader(view_position=VIEW_POSITION_FOR_MODEL)
    train_ds, validation_ds, test_ds = tf_data_loader.load_datasets()

    if train_ds is None or validation_ds is None or test_ds is None:
        logger.error("falha ao carregar datasets tensorflow. abortando pipeline de modelo.")
        return

    model_builder = ModelBuilder()
    model = model_builder.build_transfer_learning_model()

    model_trainer = ModelTrainer()
    history = model_trainer.train_model(model, train_ds, validation_ds)

    if history is None:
        logger.error("treinamento do modelo falhou. abortando avaliação do modelo.")
        return

    # avaliação salva relatórios.
    model_evaluator = ModelEvaluator()
    model_evaluator.run_evaluation(test_ds)

    logger.info("--- pipeline de modelo concluído ---")

def run_evaluation_only_pipeline():
    """avalia modelo existente e salva relatórios."""
    logger.info("--- iniciando pipeline de apenas avaliação ---")
    
    tf_data_loader = TFDataLoader(view_position=VIEW_POSITION_FOR_MODEL)
    _, _, test_ds = tf_data_loader.load_datasets()

    if test_ds is None:
        logger.error("falha ao carregar dataset de teste tensorflow. abortando pipeline de apenas avaliação.")
        return

    # garante diretório de relatórios.
    os.makedirs(REPORTS_DIR, exist_ok=True)
    model_evaluator = ModelEvaluator()

    model_evaluator.run_evaluation(test_ds)
    
    logger.info("--- pipeline de apenas avaliação concluído ---")
    logger.info(f"relatórios de avaliação (ex: matriz de confusão, relatório de classificação) salvos em '{REPORTS_DIR}'.")

def main():
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline MVP")
    parser.add_argument('--data', action='store_true', help='executa o pipeline de extração de dados e processamento de imagem.')
    parser.add_argument('--analyze', action='store_true', help='executa o pipeline de análise estatística descritiva.')
    parser.add_argument('--model', action='store_true', help='executa o pipeline de treinamento e avaliação do modelo de machine learning.')
    parser.add_argument('--evaluate_only', action='store_true', help='executa apenas a avaliação do modelo no conjunto de teste e salva relatórios (não retreina).')
    parser.add_argument('--all', action='store_true', help='executa todos os pipelines (dados, análise, modelo).')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='define o nível de log.')

    args = parser.parse_args()

    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'nível de log inválido: {args.log_level}')
    logger.setLevel(numeric_level)
    logger.info(f"nível de log definido para {args.log_level}")

    if args.all:
        run_data_pipeline()
        run_analysis_pipeline()
        run_model_pipeline()
    elif args.data:
        run_data_pipeline()
    elif args.analyze:
        run_analysis_pipeline()
    elif args.model:
        run_model_pipeline()
    elif args.evaluate_only:
        run_evaluation_only_pipeline()
    else:
        parser.print_help()
        logger.info("nenhum pipeline especificado. use --data, --analyze, --model, --evaluate_only, ou --all.")

if __name__ == "__main__":
    setup_logging()
    main()