#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Вспомогательные функции для проекта.
"""

import logging
import os
import sys
from config.parameters import OUTPUT_PARAMS

def get_project_root() -> str:
    """Получение корневой директории проекта."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def ensure_output_dirs() -> None:
    """Создание директорий для вывода результатов."""
    for dir_path in OUTPUT_PARAMS.values():
        if isinstance(dir_path, str) and dir_path:
            os.makedirs(dir_path, exist_ok=True)

def setup_logging(module_name: str, level: int = logging.INFO) -> logging.Logger:
    """Настройка логирования для модуля."""
    logger = logging.getLogger(module_name)
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger