from openpyxl import load_workbook
from pandas import DataFrame, concat
import numpy as np


def read_census_data(file_name: str, sheet_name: str, sex: str) -> DataFrame:
    """A utility function to read data from UK 2011 census files.

    Parameters
    ----------
    file_name : str
        The filename, this must be an "*.xlsx" file.
    sheet_name : str
        The sheet name.
    sex: str
        A string that encodes the sex of participants, the same for the whole
        group.

    Returns
    -------
    DataFrame
        A DataFrame that contains age groups, the total number of people per
        group and their sex.
    """

    wb = load_workbook(filename=file_name)
    sheet_ranges = wb[sheet_name]

    control_population_sum = sheet_ranges[16][5].value

    age_groups_total = 20
    df_list = []

    for age_band in range(age_groups_total):
        global_i = age_band * 7 + 7

        age_groups = [str(age_band * 5 + i) for i in range(5)]

        age_total = [sheet_ranges[16][global_i + i].value for i in range(1, 6)]

        assert sum(age_total) == sheet_ranges[16][global_i].value

        df_list.append(DataFrame({'Age group': age_groups,
                                  'Sex': [sex] * 5,
                                  'Number of people': age_total
                                  }))
    df_list.append(DataFrame({'Age group': "100 and over",
                              'Sex': [sex],
                              'Number of people': [
                                  sheet_ranges[16][
                                      age_groups_total * 7 + 7].value
                              ]}))

    df = concat(df_list, copy=False, ignore_index=True)

    assert sum(df['Number of people'].to_numpy()) == control_population_sum

    df['Age group'] = df['Age group'].astype('category')
    df['Sex'] = df['Sex'].astype('category')
    df['Number of people'] = df['Number of people'].astype(np.uint32)

    return df
