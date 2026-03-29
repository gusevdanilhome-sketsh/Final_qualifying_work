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
from classifiers.random_forest import random_forest_classifier_t
from classifiers.logistic import logistic_classifier_t
from classifiers.lda import lda_classifier_t
from classifiers.bayesian import bayesian_classifier_t
from visualization.hodograph import hodograph_plotter_t
from visualization.defect_map import defect_map_plotter_t
from visualization.reports import report_generator_t
from utils.helpers import setup_logging, get_project_root, ensure_output_dirs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from config.parameters import (
    OUTPUT_PARAMS, DATASET_PARAMS, CLASS_NAMES, 
    NOISE_PARAMS, SEGMENT_SEQUENCE
)

def main():
    """Основная функция выполнения программы."""
    # Настройка логирования
    logger = setup_logging("main")
    logger.info("Инициализация системы генерации данных и классификации...")
    
    # Создание директорий для вывода
    ensure_output_dirs()
    
    try:
        # 1. Инициализация генератора с фиксированным seed
        logger.info(f"Инициализация генератора данных (seed={NOISE_PARAMS['seed']})...")
        generator = data_generator_t(seed=NOISE_PARAMS['seed'])
        
        # 2. Генерация данных
        logger.info("Генерация обучающей выборки...")
        logger.info("Последовательность сегментов согласно таблице:")
        for i, cls in enumerate(SEGMENT_SEQUENCE):
            logger.info(f"  Сегмент {i+1} (1мм): Класс {cls} ({CLASS_NAMES[cls]})")
        
        df = generator.generate_dataset()
        
        # 3. Сохранение данных в CSV
        storage = storage_t()
        csv_filename = os.path.join(OUTPUT_PARAMS['data_dir'], "mpl_defect_dataset.csv")
        storage.save_to_csv(df, csv_filename)
        logger.info(f"Данные сохранены в файл: {csv_filename}")
        logger.info(f"Всего образцов: {len(df)}")
        
        # 4. Подготовка данных для обучения
        metadata_cols = ['label', 'defect_position', 'defect_severity', 'segment_index', 'position']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        X = df[feature_cols].values
        y = df['label'].values
        
        # Статистика по классам
        logger.info("\nРаспределение образцов по классам:")
        for cls in range(DATASET_PARAMS['classes']):
            count = np.sum(y == cls)
            logger.info(f"  Класс {cls} ({CLASS_NAMES[cls]}): {count} образцов ({count/len(y)*100:.1f}%)")
        
        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=DATASET_PARAMS['test_size'], 
            random_state=DATASET_PARAMS['random_state'], 
            stratify=y
        )
        
        logger.info(f"\nРазмер обучающей выборки: {len(X_train)}")
        logger.info(f"Размер тестовой выборки: {len(X_test)}")
        
        # 5. Обучение и верификация классификаторов
        logger.info("\nЗапуск обучения классификаторов...")
        classifiers = [
            random_forest_classifier_t(),
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
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Лучшая точность: {best_accuracy:.4f}")
        logger.info(f"Лучший классификатор: {best_classifier.get_name()}")
        logger.info(f"{'='*60}")
        
        # Детальный отчёт по лучшему классификатору
        logger.info("\nДетальный отчёт по классам:")
        logger.info(classification_report(y_test, best_y_pred, target_names=CLASS_NAMES))
        
        # 6. Генерация визуализаций
        logger.info("Генерация визуализаций...")
        
        hodograph_plotter = hodograph_plotter_t()
        hodograph_paths = hodograph_plotter.plot_hodographs_by_frequency(X_test, y_test)
        logger.info(f"Создано годографов: {len(hodograph_paths)}")
        
        hodograph_plotter.plot_combined_hodograph(X_test, y_test)
        hodograph_plotter.plot_confusion_matrix(confusion_matrix(y_test, best_y_pred))
        
        defect_map_plotter = defect_map_plotter_t()
        defect_map_paths = defect_map_plotter.plot_defect_maps_by_frequency(df, y_test, best_y_pred)
        logger.info(f"Создано карт дефектов: {len(defect_map_paths)}")
        
        defect_map_plotter.plot_defect_map_combined(df, y_test, best_y_pred)
        defect_map_plotter.plot_segment_sequence(df)
        defect_map_plotter.plot_prediction_accuracy(y_test, best_y_pred)
        
        # 7. Генерация отчётов
        logger.info("Генерация отчётов...")
        report_generator = report_generator_t()
        report_generator.generate_classification_report(y_test, best_y_pred, best_classifier.get_name())
        report_generator.generate_config_report()
        
        logger.info("\nПрограмма завершила работу успешно.")
        logger.info(f"Результаты сохранены в директорию: {OUTPUT_PARAMS['base_dir']}")
        
        # Проверка достижения целевой точности
        if best_accuracy >= 0.8:
            logger.info("✓ Целевая точность (≥0.8) достигнута!")
        else:
            logger.warning(f"⚠ Целевая точность (≥0.8) не достигнута. Текущая: {best_accuracy:.4f}")
        
    except Exception as e:
        logger.error(f"Критическая ошибка выполнения: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()