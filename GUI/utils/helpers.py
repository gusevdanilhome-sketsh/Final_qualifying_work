#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Вспомогательные функции для проекта.
"""

import logging
import os
import sys

def get_project_root() -> str:
    """
    Получение корневой директории проекта.
    
    Returns:
        str: Путь к корневой директории.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def setup_logging(module_name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Настройка логирования для модуля.
    
    Args:
        module_name: Имя модуля для логирования.
        level: Уровень логирования.
        
    Returns:
        logging.Logger: Настроенный объект логгера.
    """
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