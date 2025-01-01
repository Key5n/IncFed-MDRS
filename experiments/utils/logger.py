import logging.config


def init_logger(filename: str):
    logging.config.dictConfig(
        {
            "version": 1,
            "handlers": {
                "file": {
                    "class": "logging.FileHandler",  # Handler class for writing logs to a file
                    "filename": filename,  # Log file name
                    "level": "INFO",
                },
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": [
                    "file",
                    "console",
                ],
            },
        }
    )
