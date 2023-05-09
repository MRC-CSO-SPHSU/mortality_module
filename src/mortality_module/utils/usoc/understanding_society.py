import os
import pandas as pd


def convert_stata_csv(path_: str = None, prefix_: str = 'a_') -> None:
    """ Converts binary .dta STATA files to regular .csv files.

    This is a helper function to convert Understanding Society .dta records - as
    the most complete ones - to regular .csv files. The method is designed to
    work with one wave only.

    Parameters
    ----------
        path_ : str
            The path to a directory containing .dta files.
        prefix_ : str
            The wave prefix.
    """
    if path_ is None:
        print('No valid path specified.')
    elif not isinstance(path_, str):
        print('Provided path is not a string.')
    else:
        file_names = os.listdir(path_)
        for filename_ in file_names:
            df = pd.read_stata(path_ + filename_)
            df.columns = df.columns.str.lower().str.removeprefix(prefix_)
            df.to_csv(filename_.
                      removesuffix('.dta').
                      removeprefix(prefix_) + '.csv', index=False)
