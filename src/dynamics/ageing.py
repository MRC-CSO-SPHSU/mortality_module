import typing

from numpy import uint32
from pandas import DataFrame

from dynamics.validator import validate_column_names, validate_column_dtypes, \
    validate_age_groups, validate_sexes, validate_size
from utils.constants import SEX_LABELS


def ageing(df: DataFrame,
           birth_number: typing.Dict[str, uint32]) -> DataFrame:
    """Ages every age group by one year.

    Effectively shifts the data by 1 year, replacing the first row with the
    number of newborns. In addition, the last age group that is `100 and over`
    is a cumulative variable and needs to be updated separately.

    :param df:
    :param birth_number:
    :return:
    """

    validate_column_names(df)
    validate_column_dtypes(df)
    validate_age_groups(df)
    validate_sexes(df)
    validate_size(df)

    # todo validate birth number

    for s_value in SEX_LABELS:
        old_elderly_value = df.loc[df["Age group"] == "100 and over",
                                   df["Sex"] == s_value, "Number of people"]

        df.loc[df["Sex"] == s_value] = df.loc[
            df["Sex"] == s_value].shift(1, fill_value=birth_number[s_value])

        df.loc[df["Age group"] == "100 and over",
               df["Sex"] == s_value, "Number of people"] += old_elderly_value

    return df
