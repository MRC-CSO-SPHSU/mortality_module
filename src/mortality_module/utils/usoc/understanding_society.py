import os
from typing import Final, List, Annotated, Dict

import pandas as pd

# TODO cross-platform paths

USOC_FIELDS: Final[Annotated[List[str], 61]] = [
    'hidp', 'pidp', 'ppid', 'fnspid', 'mnspid', 'gor_dv', 'indinus_xw',
    'hhdenus_xw', 'pns1pid', 'pns2pid', 'depchl_dv', 'mastat_dv', 'jbstat',
    'hiqual_dv', 'maedqf', 'paedqf', 'dep', 'dnc', 'fimnpen_dv', 'fimnmisc_dv',
    'fiyrinvinc_dv', 'ficode', 'ded', 'drtren', 'sedcsmpl', 'istrtdaty',
    'hsownd', 'sex', 'age', 'scghqa', 'scghqb', 'scghqc', 'scghqd', 'scghqe',
    'scghqf', 'scghqg', 'scghqh', 'scghqi', 'scghqj', 'scghqk', 'scghql',
    'fimnsben_dv', 'paygu_dv', 'seearngrs_dv', 'j2pay_dv', 'sf1', 'sf2a',
    'sf2b', 'sf3a', 'sf3b', 'sf4a', 'sf4b', 'sf5', 'sf6a', 'sf6b', 'sf6c',
    'sf7', 'scflag_dv', 'jbft_dv', 'sclfsato', 'finnow']

USOC_FIELDS_SPLIT: Final[Annotated[Dict[str, List[str]], 6]] = {
    'indall': ['hidp', 'pidp', 'ppid', 'fnspid', 'mnspid', 'gor_dv', 'pns1pid',
               'pns2pid', 'depchl_dv', 'mastat_dv', 'sex', 'scflag_dv'],
    'indresp': ['hidp', 'pidp', 'ppid', 'fnspid', 'mnspid', 'indinus_xw',
                'pns1pid', 'pns2pid', 'depchl_dv', 'mastat_dv', 'jbstat',
                'hiqual_dv', 'maedqf', 'paedqf', 'fimnpen_dv', 'fimnmisc_dv',
                'fiyrinvinc_dv', 'istrtdaty', 'sex', 'scghqa', 'scghqb',
                'scghqc', 'scghqd', 'scghqe', 'scghqf', 'scghqg', 'scghqh',
                'scghqi', 'scghqj', 'scghqk', 'scghql', 'fimnsben_dv',
                'paygu_dv', 'seearngrs_dv', 'j2pay_dv', 'sf1', 'sf2a', 'sf2b',
                'sf3a', 'sf3b', 'sf4a', 'sf4b', 'sf5', 'sf6a', 'sf6b', 'sf6c',
                'sf7', 'scflag_dv', 'jbft_dv', 'sclfsato', 'finnow'],
    'hhresp': ['hidp', 'hhdenus_xw', 'hsownd'],
    'youth': ['hidp', 'pidp', 'fnspid', 'mnspid', 'pns1pid', 'pns2pid', 'sex'],
    'child': ['hidp', 'pidp', 'fnspid', 'mnspid', 'pns1pid', 'pns2pid',
              'depchl_dv', 'sex'],
    'income': ['hidp', 'pidp', 'ficode']}


def _read_usoc_fields(path_: str = None,
                      verbose: bool = True) -> dict[pd.DataFrame] | None:
    """ Reads all data from provided .csv files and keeps only what's needed.

    This is a brute-force method that loops over all .csv files in a directory,
    loads them into memory one-by-one and keeps only the columns that are in the
    USOC_FIELDS list.

    Parameters
    ----------
        path_ : str
            The path to a directory containing .csv files, `None` by default.
        verbose : bool
            A boolean flag that allows to print out filenames and corresponding
            columns, `True` by default.

    Notes
    -----
        This method is inherently unsafe, however, that allows it to be
        extremely flexible.

    Returns
    -------
    dict[pd.DataFrame] or None
        A dictionary with corresponding dataframes or `None` if `path_` is
         `None`.

    """
    if path_ is None:
        print('No valid path provided.')
    else:
        data = {}
        for filename_ in os.listdir(path_):
            if filename_.endswith('.csv'):
                df = pd.read_csv(path_ + filename_, low_memory=False)
                final_columns = [column_ for column_ in
                                 USOC_FIELDS if column_ in df.columns]
                data[filename_.split('.')[0].split('/')[-1]] = df[final_columns]

        for k, v in data.items():
            if k != 'indall':
                data[k] = v.drop(columns='gor_dv', errors='ignore')
                # drop gor_dv as redundant

        if verbose:
            for k, v in data.items():
                print(k)
                print('\t', [column_ for column_ in
                             USOC_FIELDS if column_ in v.columns])

        return data


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
