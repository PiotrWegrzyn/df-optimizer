from optimizer.series_optimizer import SeriesOptimizer


class TextSeriesOptimizer(SeriesOptimizer):
    def _get_optimized_type(self):
        return "category"

    def _get_max_size(self) -> int:
        return 0
