import os
import pandas as pd

_DATASETS_DIR = os.path.join(os.path.dirname(__file__), 'datasets')

_AVAILABLE = {
    'stroke': 'stroke.csv',
}


def load_dataset(name):
    """
    Load a sample dataset bundled with propensio.

    Parameters
    ----------
    name : str
        Name of the dataset. Available: 'stroke'

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    Load the stroke dataset for use with PropensityScoreMatch:

        import propensio

        df = propensio.load_dataset('stroke')
        print(df.head())

    """
    if name not in _AVAILABLE:
        raise ValueError(
            "Dataset '{}' not found. Available datasets: {}".format(
                name, list(_AVAILABLE.keys())
            )
        )

    path = os.path.join(_DATASETS_DIR, _AVAILABLE[name])
    return pd.read_csv(path)
