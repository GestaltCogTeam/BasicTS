import logging
from easytorch.utils.logging import logger_initialized


def clear_loggers():
    for logger_name in logger_initialized:
        # logging.getLogger(logger_name).handlers.clear()
        logger = logging.getLogger(logger_name)
        # disable the logger
        # logger.disabled = True
        # remove handlers
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()
    logger_initialized.clear()
