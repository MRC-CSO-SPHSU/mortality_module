from pathlib import Path
from typing import Final, List, Annotated, Dict
from sklearn.preprocessing import MultiLabelBinarizer

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

FICODE: Final[Annotated[Dict[str, int], 36]] = {
    'ni_retirement_state_retirement_old_age_pension': 1,
    'a_pension_from_a_previous_employer': 2,
    'a_pension_from_a_spouses_previous_employer': 3,
    'a_private_pension_annuity': 4,
    'a_widows_or_war_widows_pension': 5,
    'a_widowed_mothers_allowance_widowed_parents_allowance_bereavement_allowance': 6,
    'pension_credit_includes_guarantee_credit_and_saving_credit': 7,
    'severe_disablement_allowance': 8,
    'industrial_injury_disablement_allowance': 9,
    'disability_living_allowance': 10,
    'attendance_allowance': 11,
    'carers_allowance_formerly_invalid_care_allowance': 12,
    'war_disablement_pension': 13,
    'incapacity_benefit': 14,
    'income_support': 15,
    'job_seekers_allowance': 16,
    'national_insurance_credits': 17,
    'child_benefit_including_lone_parent_child_benefit_payments': 18,
    'child_tax_credit': 19,
    'working_tax_credit_includes_disabled_persons_tax_credit': 20,
    'maternity_allowance': 21,
    'housing_benefit': 22,
    'council_tax_benefit': 23,
    'educational_grant_not_student_loan_or_tuition_fee_loan': 24,
    'trade_union_friendly_society_payment': 25,
    'maintenance_or_alimony': 26,
    'payments_from_a_family_member_not_living_here': 27,
    'rent_from_boarders_or_lodgers_not_family_members_living_here_with_you': 28,
    'rent_from_any_other_property': 29,
    'foster_allowance_guardian_allowance': 30,
    'rent_rebate': 31,
    'rate_rebate': 32,
    'employment_and_support_allowance': 33,
    'return_to_work_credit': 34,
    'sickness_and_accident_insurance': 35,
    'in_work_credit_for_lone_parents': 36}


def merge_usoc_data(path_: str | Path = None) -> pd.DataFrame:
    """ Combines all provided USoc files into a single table of individuals.

    A raw data pre-processor, this method merges multiple tables together using
    household and person ids as keys. In the case when field values differ
    individual responses take priority.

    Parameters
    ----------
        path_ : str or Path
            The path to a directory containing .csv files, `None` by default.

    Returns
    -------
    pd.DataFrame
        A single dataframe with all required individual data.

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

    for i in range(1, len(FICODE) + 1):
        data[f'ficode_{i}'] = data[f'ficode_{i}'].fillna(0).astype(bool)

    household_response = load_file('hhresp')
    data = (data.merge(household_response,
                       how='left',
                       left_on=['hidp'],
                       right_on=['hidp'])
            )
    return data


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
