#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Точка входа в программу.
Организует процесс генерации данных, обучения и верификации классификаторов.
"""

import sys
import os

# Добавление корня проекта в путь для импортов
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.generator import data_generator_t
from data_processing.storage import storage_t
from classifiers.bayesian import bayesian_classifier_t
from classifiers.lda import lda_classifier_t
from classifiers.logistic import logistic_classifier_t
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils.helpers import setup_logging, get_project_root

def main():
    """Основная функция выполнения программы."""
    # Настройка логирования
    logger = setup_logging("main")
    logger.info("Инициализация системы генерации данных и классификации...")
    
    try:
        # 1. Инициализация генератора
        generator = data_generator_t()
        
        # 2. Генерация данных
        logger.info("Генерация обучающей выборки...")
        df = generator.generate_dataset()
        
        # 3. Сохранение в CSV
        storage = storage_t()
        csv_filename = os.path.join(get_project_root(), "mpl_defect_dataset.csv")
        storage.save_to_csv(df, csv_filename)
        logger.info(f"Данные сохранены в файл: {csv_filename}")
        
        # 4. Обучение и верификация классификаторов
        logger.info("Запуск обучения классификаторов...")
        X = df.iloc[:, :-1].values
        y = df['label'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        classifiers = [
            bayesian_classifier_t(),
            lda_classifier_t(),
            logistic_classifier_t()
        ]
        
        for clf in classifiers:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            logger.info(f"\nМодель: {clf.get_name()}")
            logger.info(classification_report(y_test, y_pred))
                
        logger.info("Программа завершила работу успешно.")
        
    except Exception as e:
        logger.error(f"Критическая ошибка выполнения: {e}")
        raise

if __name__ == "__main__":
    main()