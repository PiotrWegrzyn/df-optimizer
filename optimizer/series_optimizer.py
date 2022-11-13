from abc import ABC, abstractmethod

import pandas as pd


class SeriesOptimizer(ABC):
    def __init__(self, series: pd.Series):
        self.series = series

    def optimize(self, series: pd.Series) -> pd.Series:
        better_type = self._get_optimized_type()
        return series.astype(better_type)

    @abstractmethod
    def _get_optimized_type(self):
        raise NotImplementedError

    @abstractmethod
    def _get_max_size(self) -> int:
        raise NotImplementedError
