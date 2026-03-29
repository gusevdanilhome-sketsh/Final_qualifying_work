#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Основной модуль запуска системы контроля МПЛ.
Автор: Danil
"""

import sys
import os
import numpy as np

# Добавление корня проекта в путь для импортов
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processing.generator import data_generator_t
from data_processing.storage import storage_t
from classifiers.logistic import logistic_classifier_t
from classifiers.lda import lda_classifier_t
from classifiers.bayesian import bayesian_classifier_t
from classifiers.random_forest import random_forest_classifier_t  # Новый импорт
from visualization.hodograph import hodograph_plotter_t
from visualization.defect_map import defect_map_plotter_t
from visualization.reports import report_generator_t
from utils.helpers import setup_logging, get_project_root, ensure_output_dirs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from config.parameters import OUTPUT_PARAMS, DATASET_PARAMS

def main():
    """Основная функция выполнения программы."""
    # Настройка логирования
    logger = setup_logging("main")
    logger.info("Инициализация системы генерации данных и классификации...")
    
    # Создание директорий для вывода
    ensure_output_dirs()
    
    try:
        # 1. Инициализация генератора (с фиксированным seed)
        logger.info("Инициализация генератора данных...")
        generator = data_generator_t()
        
        # 2. Генерация данных
        logger.info("Генерация обучающей выборки...")
        df = generator.generate_dataset()
        
        # 3. Сохранение данных в CSV
        storage = storage_t()
        csv_filename = os.path.join(OUTPUT_PARAMS['data_dir'], "mpl_defect_dataset.csv")
        storage.save_to_csv(df, csv_filename)
        logger.info(f"Данные сохранены в файл: {csv_filename}")
        
        # 4. Подготовка данных для обучения
        X = df.iloc[:, :-3].values  # Исключаем label, position, severity
        y = df['label'].values
        
        # Разделение на обучающую и тестовую выборки (с фиксированным random_state)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=DATASET_PARAMS['test_size'], 
            random_state=DATASET_PARAMS['random_state'], 
            stratify=y
        )
        
        logger.info(f"Размер обучающей выборки: {len(X_train)}")
        logger.info(f"Размер тестовой выборки: {len(X_test)}")
        
        # 5. Обучение и верификация классификаторов
        logger.info("Запуск обучения классификаторов...")
        classifiers = [
            random_forest_classifier_t(),  # Добавлен Random Forest
            logistic_classifier_t(),
            lda_classifier_t(),
            bayesian_classifier_t()
        ]
        
        best_accuracy = 0.0
        best_classifier = None
        best_y_pred = None
        
        for clf in classifiers:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            
            logger.info(f"\nМодель: {clf.get_name()}")
            logger.info(f"Точность (Accuracy): {accuracy:.4f}")
            logger.info(f"Макро-F1: {macro_f1:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_classifier = clf
                best_y_pred = y_pred
        
        logger.info(f"\nЛучшая точность: {best_accuracy:.4f}")
        
        # 6. Генерация визуализаций
        logger.info("Генерация визуализаций...")
        hodograph_plotter = hodograph_plotter_t()
        hodograph_plotter.plot_hodograph(df, best_classifier, X_test, y_test)
        hodograph_plotter.plot_confusion_matrix(confusion_matrix(y_test, best_y_pred))
        
        defect_map_plotter = defect_map_plotter_t()
        defect_map_plotter.plot_defect_map(df, y_test, best_y_pred)
        defect_map_plotter.plot_prediction_accuracy(y_test, best_y_pred)
        
        # 7. Генерация отчётов
        logger.info("Генерация отчётов...")
        report_generator = report_generator_t()
        report_generator.generate_classification_report(y_test, best_y_pred, best_classifier.get_name())
        report_generator.generate_config_report()
        
        logger.info("Программа завершила работу успешно.")
        logger.info(f"Результаты сохранены в директорию: {OUTPUT_PARAMS['base_dir']}")
        
        # Проверка достижения целевой точности
        if best_accuracy >= 0.8:
            logger.info("✓ Целевая точность (≥0.8) достигнута!")
        else:
            logger.warning(f"⚠ Целевая точность (≥0.8) не достигнута. Текущая: {best_accuracy:.4f}")
        
    except Exception as e:
        logger.error(f"Критическая ошибка выполнения: {e}")
        raise

if __name__ == "__main__":
    main()