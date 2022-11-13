from typing import Union

import numpy as np
import pandas as pd

from optimizer.series_optimizer import SeriesOptimizer


class UIntSeriesOptimizer(SeriesOptimizer):
    def _get_max_size(self) -> int:
        return max(abs(self.series.min()), self.series.max())

    def _get_optimized_type(self) -> np.dtype:
        max_value = max((abs(self.series.min())-1, self.series.max()))
        if max_value <= 255:
            size = 8
        elif max_value <= 65535:
            size = 16
        elif max_value <= 4294967295:
            size = 32
        else:
            return self.series.dtype
        return np.dtype(f'int{size}')
