from pathlib import Path
from typing import Final, List, Annotated, Dict

import pandas as pd

from mortality_module.utils.utils import path_validation

USOC_FIELDS: Final[Annotated[List[str], 56]] = [
    'hidp', 'pidp', 'ppid', 'fnspid', 'mnspid', 'gor_dv', 'indinus_xw',
    'hhdenus_xw', 'pns1pid', 'pns2pid', 'depchl_dv', 'mastat_dv', 'jbstat',
    'hiqual_dv', 'maedqf', 'paedqf', 'fimnpen_dv', 'fimnmisc_dv',
    'fiyrinvinc_dv', 'ficode', 'istrtdaty', 'hsownd', 'sex_dv', 'age_dv',
    'scghqa', 'scghqb', 'scghqc', 'scghqd', 'scghqe', 'scghqf', 'scghqg',
    'scghqh', 'scghqi', 'scghqj', 'scghqk', 'scghql', 'fimnsben_dv', 'paygu_dv',
    'seearngrs_dv', 'j2pay_dv', 'sf1', 'sf2a', 'sf2b', 'sf3a', 'sf3b', 'sf4a',
    'sf4b', 'sf5', 'sf6a', 'sf6b', 'sf6c', 'sf7', 'scflag_dv', 'jbft_dv',
    'sclfsato', 'finnow']

USOC_NAME_MAP: Final[Annotated[Dict[str, List[str]], 6]] = {
    'hidp': 'id_household',
    'pidp': 'id_person',
    'ppid': 'id_partner',
    'fnspid': 'id_father',
    'mnspid': 'id_mother',
    'pns1pid': 'id_parent1',
    'pns2pid': 'id_parent2',
    'gor_dv': 'constituent_country',
    'indinus_xw': 'weight_person',
    'hhdenus_xw': 'weight_household',
    'depchl_dv': 'indicator_dependent_child',
    'mastat_dv': 'label_marital_status',
    'jbstat': 'label_job_status',
    'hiqual_dv': 'label_highest_qualification',
    'maedqf': 'label_mother_education',
    'paedqf': 'label_father_education',
    'fimnpen_dv': 'income_pension',
    'fimnmisc_dv': 'income_miscellaneous',
    'fiyrinvinc_dv': 'income_investment',
    'fimnsben_dv': 'income_benefits',
    'paygu_dv': 'income_pay',
    'seearngrs_dv': 'income_self_employment',
    'j2pay_dv': 'income_second_job',
    'ficode': 'label_income',
    'istrtdaty': 'year',
    'hsownd': 'label_house_ownership',
    'sex_dv': 'sex',
    'age_dv': 'age',
    'scghqa': 'label_ghq_a',
    'scghqb': 'label_ghq_b',
    'scghqc': 'label_ghq_c',
    'scghqd': 'label_ghq_d',
    'scghqe': 'label_ghq_e',
    'scghqf': 'label_ghq_f',
    'scghqg': 'label_ghq_g',
    'scghqh': 'label_ghq_h',
    'scghqi': 'label_ghq_i',
    'scghqj': 'label_ghq_j',
    'scghqk': 'label_ghq_k',
    'scghql': 'label_ghq_l',
    'sf1': 'label_general_health',
    'sf2a': 'label_health_limits_a',
    'sf2b': 'label_health_limits_b',
    'sf3a': 'label_health_limits_c',
    'sf3b': 'label_health_limits_d',
    'sf4a': 'label_emotions_a',
    'sf4b': 'label_emotions_b',
    'sf5': 'label_pain',
    'sf6a': 'label_feel_calm',
    'sf6b': 'label_have_energy',
    'sf6c': 'label_feel_depressed',
    'sf7': 'label_social_activities_interference',
    'scflag_dv': 'indicator_self_completion',
    'jbft_dv': 'indicator_full_part_time',
    'sclfsato': 'label_life_satisfaction',
    'finnow': 'label_financial_situation'}

USOC_FIELDS_SPLIT: Final[Annotated[Dict[str, List[str]], 6]] = {
    'indall': ['hidp', 'pidp', 'ppid', 'fnspid', 'mnspid', 'gor_dv', 'pns1pid',
               'pns2pid', 'depchl_dv', 'mastat_dv', 'sex_dv', 'scflag_dv'],
    'indresp': ['hidp', 'pidp', 'ppid', 'fnspid', 'mnspid', 'indinus_xw',
                'pns1pid', 'pns2pid', 'depchl_dv', 'mastat_dv', 'jbstat',
                'hiqual_dv', 'maedqf', 'paedqf', 'fimnpen_dv', 'fimnmisc_dv',
                'fiyrinvinc_dv', 'istrtdaty', 'sex_dv', 'scghqa', 'scghqb',
                'scghqc', 'scghqd', 'scghqe', 'scghqf', 'scghqg', 'scghqh',
                'scghqi', 'scghqj', 'scghqk', 'scghql', 'fimnsben_dv',
                'paygu_dv', 'seearngrs_dv', 'j2pay_dv', 'sf1', 'sf2a', 'sf2b',
                'sf3a', 'sf3b', 'sf4a', 'sf4b', 'sf5', 'sf6a', 'sf6b', 'sf6c',
                'sf7', 'scflag_dv', 'jbft_dv', 'sclfsato', 'finnow'],
    'hhresp': ['hidp', 'hhdenus_xw', 'hsownd'],
    'youth': ['hidp', 'pidp', 'fnspid', 'mnspid', 'pns1pid', 'pns2pid',
              'sex_dv'],
    'child': ['hidp', 'pidp', 'fnspid', 'mnspid', 'pns1pid', 'pns2pid',
              'depchl_dv', 'sex_dv'],
    'income': ['hidp', 'pidp', 'ficode']}


def _read_usoc_fields(path_: str | Path = None,
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
        A dictionary with corresponding dataframes.

    """
    p_ = path_validation(path_)

    data = {}
    for filename_ in [x for x in p_.iterdir()]:
        if filename_.suffix == '.csv':
            df = pd.read_csv(path_ / filename_, low_memory=False)
            final_columns = [column_ for column_ in
                             USOC_FIELDS if column_ in df.columns]
            data[filename_.stem] = df[final_columns]

    for k, v in data.items():
        if k != 'indall':
            data[k] = v.drop(columns='gor_dv', errors='ignore')

    if verbose:
        for k, v in data.items():
            print(k)
            print('\t', [column_ for column_ in
                         USOC_FIELDS if column_ in v.columns])

    return data


def convert_stata_csv(path_: str | Path = None, prefix_: str = 'a_') -> None:
    """ Converts binary .dta STATA files to regular .csv files.

    This is a helper function to convert Understanding Society .dta records - as
    the most complete ones - to regular .csv files. The method is designed to
    work with one wave only. All output data is stored in the same directory.

    Parameters
    ----------
        path_ : str
            The path to a directory containing .dta files.
        prefix_ : str
            The wave prefix.
    """
    p_ = path_validation(path_)

    for filename_ in [x for x in p_.iterdir()]:
        df = pd.read_stata(path_ / filename_)
        df.columns = df.columns.str.lower().str.removeprefix(prefix_)
        df.to_csv(path_ / Path(filename_.
                               name.
                               removesuffix('.dta').
                               removeprefix(prefix_).
                               join(['.csv']), index=False))


def _find_dangling_fields(path_: str | Path = None) -> set:
    """Ensures that all requested fields are actually in USoc datasets.

    Parameters
    ----------
        path_ : str
            The path to a directory containing .csv files, `None` by default.

    Returns
    -------
    set
        An empty set when everything is correct or a set of invalid column names
         otherwise.

    """
    actual_fields = []
    for k, v in _read_usoc_fields(path_, False).items():
        actual_fields += [column_ for column_ in
                          USOC_FIELDS if column_ in v.columns]
    actual_fields = set(actual_fields)
    return actual_fields.symmetric_difference(USOC_FIELDS)
