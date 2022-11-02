"""
Logger config module.
"""

import logging


def set_logger(cache_path: str) -> None:
    """
    Set logger config;
    :param cache_path: path to logging file;
    :return: None.
    """
    logging.basicConfig(
        level=logging.INFO,
        filename=cache_path,
        filemode="a+",
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
