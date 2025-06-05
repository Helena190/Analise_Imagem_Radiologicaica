import logging
import os

def setup_logging(log_file='app.log', level=logging.INFO):
    """configura o registro (logging) para a aplicação."""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filepath = os.path.join(log_dir, log_file)

    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_filepath)

    c_handler.setLevel(level)
    f_handler.setLevel(level)

    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

logger = setup_logging()

if __name__ == "__main__":
    logger.info("esta é uma mensagem de informação de logger.py")
    logger.warning("esta é uma mensagem de aviso de logger.py")
    logger.error("esta é uma mensagem de erro de logger.py")