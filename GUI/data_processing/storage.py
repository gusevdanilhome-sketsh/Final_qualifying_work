#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для сохранения и загрузки данных.
"""

import pandas as pd

class storage_t:
    """Управление сохранением данных в файлы."""
    
    def save_to_csv(self, df: pd.DataFrame, filename: str) -> None:
        df.to_csv(filename, index=False)

    def load_from_csv(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(filename)