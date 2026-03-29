#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Основной модуль запуска системы контроля МПЛ.
Автор: Danil
"""

import sys
import os

# Добавление корня проекта в путь для импортов
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing.generator import data_generator_t
from data_processing.storage import storage_t
from classifiers.bayesian import bayesian_classifier_t
from classifiers.lda import lda_classifier_t
from classifiers.logistic import logistic_classifier_t
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def main():
    """Основная функция выполнения программы."""
    print("Инициализация системы генерации данных и классификации...")
    
    # 1. Инициализация генератора
    generator = data_generator_t()
    
    # 2. Генерация данных
    print("Генерация обучающей выборки...")
    df = generator.generate_dataset()
    
    # 3. Сохранение в CSV
    storage = storage_t()
    csv_filename = "mpl_defect_dataset.csv"
    storage.save_to_csv(df, csv_filename)
    print(f"Данные сохранены в файл: {csv_filename}")
    
    # 4. Обучение и верификация классификаторов
    print("\nЗапуск обучения классификаторов...")
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
        
        print(f"\nМодель: {clf.get_name()}")
        print(classification_report(y_test, y_pred))
            
    print("\nПрограмма завершила работу успешно.")

if __name__ == "__main__":
    main()