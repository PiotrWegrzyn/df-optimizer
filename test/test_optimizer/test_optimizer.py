import numpy as np
import pandas as pd
import pytest
from optimizer.dataframe_optimizer import DataframeOptimizer

@pytest.mark.parametrize(
    "values,expected_type",
    [
        ([-2**7], np.dtype("int8")),
        ([-2**15], np.dtype("int16")),
        ([-2**31], np.dtype("int32")),
        ([-2**63], np.dtype("int64")),
        ([2**8], np.dtype("uint8")),
        ([2**16], np.dtype("uint16")),
        ([2**32], np.dtype("uint32")),
        ([2**64], np.dtype("uint64")),
        (["abc"], pd.DataFrame(["abc"], columns=['data'])["data"].astype("category").dtype)
    ]
)
def test_numeric_values(values, expected_type):
    df = pd.DataFrame(values, columns=['data'])
    optimizer = DataframeOptimizer(df)
    optimizer.optimize()
    assert df["data"].dtype == expected_type
