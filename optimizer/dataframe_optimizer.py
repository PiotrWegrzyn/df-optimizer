import numpy as np
import pandas as pd

from optimizer.float_series_optimizer import FloatSeriesOptimizer
from optimizer.int_series_optimizer import IntSeriesOptimizer
from optimizer.series_optimizer import SeriesOptimizer
from optimizer.text_series_optimizer import TextSeriesOptimizer
from optimizer.uint_series_optimizer import UIntSeriesOptimizer


class DataframeOptimizer:
    registered_optimizers = (
        TextSeriesOptimizer,
        IntSeriesOptimizer,
        UIntSeriesOptimizer,
        FloatSeriesOptimizer
    )

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def optimize(self) -> None:
        for col_name in self.df:
            self.df[col_name] = self.optimize_series(self.df[col_name])

    def optimize_series(self, series: pd.Series) -> pd.Series:
        optimizer = self.get_series_optimizer(series)
        return optimizer.optimize(series)

    def get_series_optimizer(self, series: pd.Series) -> SeriesOptimizer:
        if series.dtype in (np.dtype('int_'), np.dtype('int64')):
            return IntSeriesOptimizer(series)
        elif series.dtype in (np.dtype('float_'), np.dtype('float64')):
            return FloatSeriesOptimizer(series)
        elif series.dtype == np.dtype('O'):
            return TextSeriesOptimizer(series)
