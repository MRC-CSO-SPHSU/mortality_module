from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from mortality_module.utils.usoc.usoc_constants import *
from mortality_module.utils.utils import path_validation


def merge_usoc_data(path_: str | Path =
                    None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merges all provided USoc data into tables for individuals and households.

    A raw data pre-processor, this method merges multiple tables together using
    household and person ids as keys. In the case when field values differ
    individual responses take priority.

    Parameters
    ----------
        path_ : str or Path
            The path to a directory containing .csv files, `None` by default.

    Returns
    -------
    tuple
        A tuple of two dataframes, one contains data per individual, the other
         one - per household.

    """
    p_ = path_validation(path_)

    def load_file(filename: str) -> pd.DataFrame:
        """Reads a .csv file and sets indices to 'pidp' and 'hidp'.

        Parameters
        ----------
        filename : str
            The name of a file.

        Returns
        -------
        pd.DataFrame
            A dataframe.
        """
        f = pd.read_csv(p_ / (filename + '.csv'),
                        usecols=USOC_FIELDS_SPLIT[filename],
                        low_memory=False
                        )
        i_ = ['pidp', 'hidp'] if {'pidp',
                                  'hidp'} <= set(f.columns) else ['hidp']
        f[i_] = f[i_].astype(int)
        return f.set_index(i_)

    data = load_file('indall')

    def find_merge_divergent_data(other_file: str) -> pd.DataFrame:
        """ A helper function to reduce copy-paste code, does the merging.

        Combines two files into one while making sure that individual responses
        take priority over the same fields but with different values.


        Parameters
        ----------
        other_file : str
            The name of a file

        Returns
        -------
        pd.DataFrame
            The result of merging and override.

        """
        new_df = load_file(other_file)

        common_fields = list(set(data.columns) & set(new_df.columns))
        for f_ in common_fields:
            mask_ = (data.loc[new_df.index, f_] == new_df[f_])
            if not all(mask_):
                print(f'Individual responses take precedence, replacing {f_}')
                not_mask_ = ~mask_
                data.loc[new_df.index, f_][not_mask_] = new_df[f_][not_mask_]

        new_fields = list(set(new_df.columns) - set(common_fields))
        data.loc[new_df.index, new_fields] = new_df[new_fields]
        if len(new_fields) > 0:
            data.loc[new_df.index, new_fields] = new_df[new_fields]

        return data

    data = find_merge_divergent_data('indresp')
    data = find_merge_divergent_data('youth')
    data = find_merge_divergent_data('child')

    income = load_file('income')
    income['ficode'] = (income['ficode']
                        .apply(lambda l: l.lower())
                        .apply(lambda l: l.replace(' / ', '/'))
                        .apply(lambda l: l.replace(' ', '_'))
                        .apply(lambda l: l.replace('-', '_'))
                        .apply(lambda l: l.replace('(', ''))
                        .apply(lambda l: l.replace(')', ''))
                        .apply(lambda l: l.replace('\'', ''))
                        .apply(lambda l: l.replace('/', '_'))
                        .apply(lambda l: l.replace('&amp;', 'and'))
                        .apply(lambda l: l.replace('__', '_'))
                        )

    income = (income
              .replace({'ficode': FICODE})
              .groupby(['hidp', 'pidp'])['ficode']
              .apply(list)
              .reset_index()
              .set_index(['hidp', 'pidp'])
              )

    mlb = MultiLabelBinarizer()
    income = pd.DataFrame(mlb.fit_transform(income.pop('ficode')),
                          columns=[f'ficode_{v}' for v in mlb.classes_],
                          index=income.index
                          )

    data = (data.merge(income,
                       how='left',
                       left_on=['pidp', 'hidp'],
                       right_on=['pidp', 'hidp'])
            )

    household_response = load_file('hhresp')

    dates = data.copy().reset_index()[['istrtdaty', 'hidp']]
    locations = data.copy().reset_index()[['gor_dv', 'hidp']]

    data = data.drop(columns=['gor_dv', 'istrtdaty'])

    def clean_household_variables(df_: pd.DataFrame, name_: str, type_):
        df_ = (df_
               .drop_duplicates()
               .groupby(['hidp'])[name_]
               .apply(list)
               .reset_index()
               )

        def _parser(list_):
            p = []
            for l_ in list_:
                try:
                    p.append(type_(l_))
                except ValueError:
                    continue
            p = set(p)
            return p.pop() if len(p) > 0 else None

        df_[name_] = df_[name_].apply(_parser)
        return df_

    dates = clean_household_variables(dates, 'istrtdaty', int)
    locations = (clean_household_variables(locations, 'gor_dv', str)
                 .apply(lambda x: x.str.strip() if x.dtype == "object" else x)
                 )

    household_response = (household_response
                          .merge(dates,
                                 how='inner',
                                 left_on=['hidp'],
                                 right_on=['hidp'])
                          .merge(locations,
                                 how='inner',
                                 left_on=['hidp'],
                                 right_on=['hidp'])
                          )

    data = data.rename(columns=USOC_NAME_MAP)
    household_response = household_response.rename(columns=USOC_NAME_MAP)
    return data, household_response


def _read_usoc_fields(path_: str | Path = None,
                      verbose: bool = True) -> dict[pd.DataFrame] | None:
    """ Reads all data from provided .csv files and keeps only what's needed.

    This is a brute-force method that loops over all .csv files in a directory,
    loads them into memory one-by-one and keeps only the columns that are in the
    USOC_FIELDS list.

    Parameters
    ----------
        path_ : str or Path
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
        path_ : str or Path
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
        path_ : str or Path
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
