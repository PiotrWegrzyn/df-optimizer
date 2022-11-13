from typing import Union

import numpy as np
import pandas as pd

from optimizer.series_optimizer import SeriesOptimizer


class IntSeriesOptimizer(SeriesOptimizer):
    def _get_max_size(self) -> int:
        return max(abs(self.series.min()), self.series.max())

    def _get_optimized_type(self) -> np.dtype:
        max_value = max((abs(self.series.min())-1, self.series.max()))
        if max_value <= 127:
            size = 8
        elif max_value <= 32767:
            size = 16
        elif max_value <= 2147483647:
            size = 32
        else:
            return self.series.dtype
        return np.dtype(f'int{size}')
