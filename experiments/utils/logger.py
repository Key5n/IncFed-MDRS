import logging


def init_logger(filename: str):
    logging.basicConfig(filename=filename, level=logging.DEBUG, filemode="w")
    # Create a logger
    logger = logging.getLogger(__name__)

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)

    # Create a stream handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
