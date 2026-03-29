#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль генерации отчётов о результатах классификации.
"""

import os
import json
import pandas as pd
from datetime import datetime
from config.parameters import OUTPUT_PARAMS
from sklearn.metrics import classification_report, accuracy_score, f1_score

class report_generator_t:
    """Класс для генерации отчётов."""
    
    def __init__(self):
        self.reports_dir = OUTPUT_PARAMS['reports_dir']
        os.makedirs(self.reports_dir, exist_ok=True)

    def generate_classification_report(self, y_true, y_pred, classifier_name: str) -> str:
        """
        Генерация текстового отчёта о классификации.
        
        Returns:
            Путь к файлу отчёта
        """
        class_names = ['Без дефекта', 'Утонение высоты', 'Утонение ширины', 
                       'Утонение подложки', 'Изменение εr']
        
        report_text = classification_report(y_true, y_pred, target_names=class_names)
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filepath = os.path.join(self.reports_dir, f'classification_report_{timestamp}.txt')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("ОТЧЁТ О КЛАССИФИКАЦИИ ДЕФЕКТОВ МПЛ\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Классификатор: {classifier_name}\n\n")
            f.write(f"Общая точность (Accuracy): {accuracy:.4f}\n")
            f.write(f"Макро-F1 Score: {macro_f1:.4f}\n\n")
            f.write("Детальный отчёт по классам:\n")
            f.write("-" * 60 + "\n")
            f.write(report_text)
            f.write("\n" + "=" * 60 + "\n")
        
        return filepath

    def generate_config_report(self) -> str:
        """Генерация отчёта о конфигурации системы."""
        from config.parameters import LINE_PARAMS, PROBE_PARAMS, FREQ_PARAMS, NOISE_PARAMS, DATASET_PARAMS
        
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filepath = os.path.join(self.reports_dir, f'config_report_{timestamp}.json')
        
        config = {
            'line_parameters': LINE_PARAMS,
            'probe_parameters': PROBE_PARAMS,
            'frequency_parameters': {
                'frequencies_hz': FREQ_PARAMS['frequencies'].tolist(),
                'power_w': FREQ_PARAMS['power']
            },
            'noise_parameters': NOISE_PARAMS,
            'dataset_parameters': DATASET_PARAMS
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        return filepath