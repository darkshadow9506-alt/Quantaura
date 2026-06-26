import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def trending_df():
    rng = np.random.default_rng(1)
    steps = rng.normal(0.0009, 0.012, 500)
    steps[120:160] += 0.01
    close = 100 * np.exp(np.cumsum(steps))
    idx = pd.date_range("2021-01-01", periods=len(close), freq="B")
    high = close * (1 + rng.uniform(0, 0.004, len(close)))
    low = close * (1 - rng.uniform(0, 0.004, len(close)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {"open": open_,
         "high": np.maximum.reduce([open_, high, close]),
         "low": np.minimum.reduce([open_, low, close]),
         "close": close,
         "volume": rng.uniform(1e6, 5e6, len(close))},
        index=idx,
    )
