#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль построения годографов коэффициента отражения.
Основано на разделе "Годограф коэффициента отражения" VKR_V2.docx.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
from config.parameters import OUTPUT_PARAMS, FREQ_PARAMS

# Попытка загрузки кириллического шрифта
def setup_fonts():
    """Настройка шрифтов для поддержки кириллицы."""
    font_paths = [
        'C:/Windows/Fonts/arial.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    ]
    for path in font_paths:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            plt.rcParams['font.family'] = 'Arial' if 'arial' in path else 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False
            return
    plt.rcParams['font.family'] = 'sans-serif'

class hodograph_plotter_t:
    """Класс для построения годографов."""
    
    def __init__(self):
        setup_fonts()
        self.figures_dir = OUTPUT_PARAMS['figures_dir']
        os.makedirs(self.figures_dir, exist_ok=True)

    def plot_hodograph(self, data_df: pd.DataFrame, classifier, X_test, y_test) -> str:
        """
        Построение годографов для каждого класса.
        
        Args:
            data_df: DataFrame с данными
            classifier: Обученный классификатор
            X_test: Тестовые признаки
            y_test: Тестовые метки
            
        Returns:
            Путь к сохранённому файлу
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Годографы коэффициента отражения по классам дефектов', fontsize=14)
        
        class_names = ['Без дефекта', 'Утонение высоты', 'Утонение ширины', 
                       'Утонение подложки', 'Изменение εr']
        colors = ['black', 'red', 'blue', 'green', 'orange']
        
        # Извлекаем комплексные коэффициенты для частот
        n_freq = len(FREQ_PARAMS['frequencies'])
        
        for cls in range(5):
            ax = axes[cls // 3, cls % 3]
            mask = y_test == cls
            
            if np.sum(mask) == 0:
                ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Класс {cls}: {class_names[cls]}')
                continue
            
            # Для демонстрации берём первые 10 образцов класса
            sample_indices = np.where(mask)[0][:10]
            
            for idx in sample_indices:
                # Формируем годограф по частотам (I/Q пары)
                sample = X_test[idx]
                real_parts = sample[0::2][:n_freq]  # I составляющие
                imag_parts = sample[1::2][:n_freq]  # Q составляющие
                
                ax.plot(real_parts, imag_parts, 'o-', alpha=0.5, color=colors[cls], markersize=4)
            
            ax.set_xlabel('I (действительная часть)')
            ax.set_ylabel('Q (мнимая часть)')
            ax.set_title(f'Класс {cls}: {class_names[cls]}')
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.axvline(0, color='gray', linewidth=0.5)
        
        # Пустой subplot скрываем
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        filepath = os.path.join(self.figures_dir, 'hodographs.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath

    def plot_confusion_matrix(self, cm: np.ndarray, class_names: list = None) -> str:
        """Построение матрицы ошибок."""
        if class_names is None:
            class_names = ['Без дефекта', 'Утонение высоты', 'Утонение ширины', 
                           'Утонение подложки', 'Изменение εr']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title='Матрица ошибок классификации',
               ylabel='Истинный класс',
               xlabel='Предсказанный класс')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Добавляем значения в ячейки
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        filepath = os.path.join(self.figures_dir, 'confusion_matrix.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath