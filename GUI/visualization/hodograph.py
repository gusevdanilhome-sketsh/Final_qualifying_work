#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль построения годографов коэффициента отражения.
Основано на разделе "Годограф коэффициента отражения" VKR_V2.docx.
Годографы разделяются по частотам, каждый тип дефекта имеет свой цвет.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
from config.parameters import OUTPUT_PARAMS, FREQ_PARAMS, CLASS_NAMES, CLASS_COLORS

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

    def plot_hodographs_by_frequency(self, X_test: np.ndarray, y_test: np.ndarray) -> list:
        """
        Построение годографов для каждой частоты отдельно.
        Каждый тип дефекта имеет свой цвет.
        
        Args:
            X_test: Тестовые признаки
            y_test: Тестовые метки
            
        Returns:
            Список путей к сохранённым файлам
        """
        n_freq = len(FREQ_PARAMS['frequencies'])
        n_features_per_freq = 6  # 3 канала × 2 компоненты (I/Q)
        filepaths = []
        
        for freq_idx in range(n_freq):
            freq = FREQ_PARAMS['frequencies'][freq_idx]
            freq_ghz = freq / 1e9
            
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.suptitle(f'Годограф коэффициента отражения на частоте {freq_ghz:.1f} ГГц', fontsize=14)
            
            # Извлечение данных для данной частоты
            start_idx = freq_idx * n_features_per_freq
            
            for cls in range(len(CLASS_NAMES)):
                mask = y_test == cls
                if np.sum(mask) == 0:
                    continue
                
                # Берём первые 20 образцов каждого класса для читаемости
                sample_indices = np.where(mask)[0][:20]
                
                for idx in sample_indices:
                    sample = X_test[idx]
                    # I/Q для трёх каналов
                    sum_i = sample[start_idx + 0]
                    sum_q = sample[start_idx + 1]
                    diff_x_i = sample[start_idx + 2]
                    diff_x_q = sample[start_idx + 3]
                    diff_y_i = sample[start_idx + 4]
                    diff_y_q = sample[start_idx + 5]
                    
                    # Используем разностный канал для годографа (более информативен)
                    ax.plot(diff_x_i, diff_x_q, 'o', alpha=0.5, 
                           color=CLASS_COLORS[cls], markersize=6,
                           label=f'{CLASS_NAMES[cls]}' if idx == sample_indices[0] else "")
            
            ax.set_xlabel('I (действительная часть), В')
            ax.set_ylabel('Q (мнимая часть), В')
            ax.set_title(f'Частота {freq_ghz:.1f} ГГц')
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.axvline(0, color='gray', linewidth=0.5)
            ax.legend(loc='upper right', fontsize=9)
            
            plt.tight_layout()
            filepath = os.path.join(self.figures_dir, f'hodograph_{freq_ghz:.1f}GHz.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            filepaths.append(filepath)
        
        return filepaths

    def plot_combined_hodograph(self, X_test: np.ndarray, y_test: np.ndarray) -> str:
        """
        Построение сводного годографа по всем частотам.
        """
        n_freq = len(FREQ_PARAMS['frequencies'])
        n_features_per_freq = 6
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle('Годографы коэффициента отражения по всем частотам', fontsize=14)
        axes = axes.flatten()
        
        for freq_idx in range(n_freq):
            freq = FREQ_PARAMS['frequencies'][freq_idx]
            freq_ghz = freq / 1e9
            ax = axes[freq_idx]
            
            start_idx = freq_idx * n_features_per_freq
            
            for cls in range(len(CLASS_NAMES)):
                mask = y_test == cls
                if np.sum(mask) == 0:
                    continue
                
                sample_indices = np.where(mask)[0][:15]
                
                for idx in sample_indices:
                    sample = X_test[idx]
                    diff_x_i = sample[start_idx + 2]
                    diff_x_q = sample[start_idx + 3]
                    
                    ax.plot(diff_x_i, diff_x_q, 'o', alpha=0.5, 
                           color=CLASS_COLORS[cls], markersize=5)
            
            ax.set_xlabel('I')
            ax.set_ylabel('Q')
            ax.set_title(f'{freq_ghz:.1f} ГГц')
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.axvline(0, color='gray', linewidth=0.5)
        
        # Скрываем пустые subplot
        for i in range(n_freq, len(axes)):
            axes[i].axis('off')
        
        # Добавляем общую легенду
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=CLASS_COLORS[i], 
                   markersize=8, label=CLASS_NAMES[i])
            for i in range(len(CLASS_NAMES))
        ]
        fig.legend(handles=legend_elements, loc='upper center', 
                   bbox_to_anchor=(0.5, 0.02), ncol=5, fontsize=10)
        
        plt.tight_layout()
        filepath = os.path.join(self.figures_dir, 'hodograph_combined.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath

    def plot_confusion_matrix(self, cm: np.ndarray, class_names: list = None) -> str:
        """Построение матрицы ошибок."""
        if class_names is None:
            class_names = CLASS_NAMES
        
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