#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль построения карт дефектов микрополосковой линии.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
from config.parameters import OUTPUT_PARAMS, LINE_PARAMS, DEFECT_PARAMS

class defect_map_plotter_t:
    """Класс для построения карт дефектов."""
    
    def __init__(self):
        self.figures_dir = OUTPUT_PARAMS['figures_dir']
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Настройка шрифтов
        font_paths = ['C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
        for path in font_paths:
            if os.path.exists(path):
                font_manager.fontManager.addfont(path)
                plt.rcParams['font.family'] = 'Arial' if 'arial' in path else 'DejaVu Sans'
                plt.rcParams['axes.unicode_minus'] = False
                return

    def plot_defect_map(self, data_df: pd.DataFrame, y_true: np.ndarray, 
                        y_pred: np.ndarray) -> str:
        """
        Построение карты дефектов вдоль линии.
        
        Args:
            data_df: DataFrame с данными (включая позиции)
            y_true: Истинные метки
            y_pred: Предсказанные метки
            
        Returns:
            Путь к сохранённому файлу
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Параметры линии
        line_length = LINE_PARAMS['length']
        line_width = LINE_PARAMS['width']
        
        # Отображение микрополосковой линии
        ax.add_patch(plt.Rectangle((0, -line_width/2), line_length, line_width, 
                                    fill=True, color='lightgray', alpha=0.5, label='МПЛ'))
        
        class_names = ['Без дефекта', 'Утонение высоты', 'Утонение ширины', 
                       'Утонение подложки', 'Изменение εr']
        colors = ['green', 'red', 'blue', 'orange', 'purple']
        
        # Отображение дефектов
        if 'defect_position' in data_df.columns:
            positions = data_df['defect_position'].values
        else:
            positions = np.random.choice(DEFECT_PARAMS['positions'], len(y_true))
        
        # Ограничиваем количество отображаемых точек для читаемости
        max_points = 100
        step = max(1, len(y_true) // max_points)
        
        for i in range(0, len(y_true), step):
            true_label = int(y_true[i])
            pred_label = int(y_pred[i])
            pos = positions[i]
            
            if true_label == 0:
                continue  # Пропускаем бездефектные
            
            # ИСПРАВЛЕНИЕ: Явное преобразование булева значения в int
            is_error = int(true_label != pred_label)
            color = 'red' if is_error else 'green'
            marker = 'x' if is_error else 'o'
            
            ax.plot(pos, 0, marker, color=color, markersize=8, alpha=0.7)
        
        ax.set_xlabel('Продольная координата, м')
        ax.set_ylabel('Поперечная координата, м')
        ax.set_title('Карта дефектов микрополосковой линии\n(Зелёный - верно, Красный - ошибка)')
        ax.set_xlim(0, line_length)
        ax.set_ylim(-line_width*2, line_width*2)
        ax.grid(True, alpha=0.3)
        
        # Добавляем легенду вручную
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Верно'),
            Line2D([0], [0], marker='x', color='w', markerfacecolor='red', markersize=10, label='Ошибка')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        filepath = os.path.join(self.figures_dir, 'defect_map.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath

    def plot_prediction_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Построение графика точности по классам."""
        from sklearn.metrics import classification_report
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        class_names = ['Без дефекта', 'Утонение высоты', 'Утонение ширины', 
                       'Утонение подложки', 'Изменение εr']
        
        # Расчёт метрик
        report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        precisions = [report_dict[name]['precision'] for name in class_names]
        recalls = [report_dict[name]['recall'] for name in class_names]
        f1_scores = [report_dict[name]['f1-score'] for name in class_names]
        
        x = np.arange(len(class_names))
        width = 0.25
        
        ax.bar(x - width, precisions, width, label='Precision', color='blue', alpha=0.7)
        ax.bar(x, recalls, width, label='Recall', color='green', alpha=0.7)
        ax.bar(x + width, f1_scores, width, label='F1-Score', color='red', alpha=0.7)
        
        ax.set_xlabel('Класс дефекта')
        ax.set_ylabel('Значение метрики')
        ax.set_title('Метрики качества классификации по классам')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        filepath = os.path.join(self.figures_dir, 'accuracy_by_class.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath