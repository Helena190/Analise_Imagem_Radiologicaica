import logging
import os

def setup_logging(log_file='app.log', level=logging.INFO):
    """
    Configura o registro (logging) para a aplicação.

    Args:
        log_file (str): O nome do arquivo de log.
        level (int): O nível de log (ex: logging.INFO, logging.DEBUG).
    """
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filepath = os.path.join(log_dir, log_file)

    # Cria um logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Limpa handlers existentes para evitar logs duplicados em Jupyter/re-execuções
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

    # Cria handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_filepath)

    # Define os níveis para os handlers
    c_handler.setLevel(level)
    f_handler.setLevel(level)

    # Cria formatadores e os adiciona aos handlers
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Adiciona handlers ao logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

# Inicializa o logger para o módulo
logger = setup_logging()

if __name__ == "__main__":
    logger.info("This is an info message from logger.py")
    logger.warning("This is a warning message from logger.py")
    logger.error("This is an error message from logger.py")