from numpy import dtype, isinf, uint32
from pandas import DataFrame, RangeIndex
import re
import warnings

from utils.constants import COLUMN_NAMES, DTYPES, AGE_GROUPS, SEX_LABELS


def validate_column_names(df: DataFrame) -> None:
    """Checks if all required columns are present.

    This functions goes through the column names of a DataFrame and checks if
    all the mandatory ones from `COLUMN_NAMES`.

    Parameters
    ----------
    df: DataFrame
        A dataframe with basic population data.

    Raises
    ------
    ValueError
        When there is a mismatch.
    """

    column_names = df.columns.values.tolist()
    for name in column_names:
        if name not in COLUMN_NAMES:
            raise ValueError(f"DataFrame doesn't contain all required columns: "
                             f"{name} is missing")


def validate_column_dtypes(df: DataFrame) -> None:
    """Ensures all required columns are of corresponding dtypes.

    It is assumed that all checked columns are present.

    Parameters
    ----------
    df: DataFrame
        A dataframe with basic population data.

    Raises
    ------
    ValueError
        When there is a mismatch.
    """

    for name in COLUMN_NAMES:
        if DTYPES[name] != df[name].dtype:
            raise ValueError(f"Columns are of invalid type: {name} must be "
                             f"{DTYPES[name]}, but is {df[name].dtype}")


def validate_age_groups(df: DataFrame) -> None:
    """Checks that all required age groups are in the DataFrame.

    Parameters
    ----------
    df: DataFrame
        A dataframe with basic population data.

    Raises
    ------
    ValueError
        When there is a discrepancy.
    """
    delta = set(df["Age group"].to_list()).difference(set(AGE_GROUPS))
    if len(delta) != 0:
        raise ValueError(f"Age groups do not cover the whole range; "
                         f"the difference is {delta}")


def validate_sexes(df: DataFrame) -> None:
    delta = set(df["Sex"].to_list()).difference(set(SEX_LABELS))
    if len(delta) != 0:
        raise ValueError(f"Sexes do not match the expected values; "
                         f"the difference is {delta}")


def validate_size(df: DataFrame) -> None:
    if df.shape[0] % len(AGE_GROUPS) != 0 | df.shape[0] % len(
            SEX_LABELS) != 0:
        raise ValueError(f"Invalid DataFrame size: {df.shape[0]} is not "
                         f"divisible by {len(AGE_GROUPS)} or {len(SEX_LABELS)}")


def validate_mortality_rates(mtr: DataFrame, mode: str) -> None:
    """Ensures that mortality rates are compliant.

    This function assumes that the mode has been validated already.


    """
    if not isinstance(mtr, DataFrame):
        raise ValueError(f"The object is not a DataFrame, but {type(mtr)}")

    if mtr.shape[0] == 0 | mtr.shape[1] == 0:
        raise ValueError(f"Invalid size {mtr.shape[0]}x{mtr.shape[1]}, "
                         f"no actual data is passed.")

    year_list = mtr.columns.values.tolist()
    r = re.compile("((19)((5)[^0]|[6-9][0-9])|(20)([0-6])(?(6)[0-9]|)|(2070))")

    if sum(1 for _ in filter(r.match, year_list)) != len(year_list):
        raise ValueError("Column names that are not in `1951`-`2070` range "
                         "are provided")

    actual_ages = mtr.index
    if not isinstance(actual_ages, RangeIndex):
        raise ValueError(f"Age index is of invalid type {type(actual_ages)}")

    if actual_ages.stop > 125 or actual_ages.step != 1:
        raise ValueError(f"Invalid range limit {actual_ages.stop} "
                         f"and stride {actual_ages.step}")

    for i in range(1, len(mtr.dtypes)):
        if mtr.dtypes[0] != mtr.dtypes[i]:
            raise ValueError(f"Inconsistent dtypes: {mtr.dtypes[0]} "
                             f"vs. {mtr.dtypes[i]}")

    if mtr.dtypes[0] not in [dtype('int64'), dtype('float64')]:
        raise ValueError(f"dtype must be {dtype('int64')} or "
                         f"{dtype('float64')}, but is {mtr.dtypes[0]}")
    if mtr.isnull().values.any():
        raise ValueError("NaN in the table.")

    if isinf(mtr).values.sum():
        raise ValueError("Inf in the table.")

    if mtr.lt(0).values.sum() > 0:
        raise ValueError("Negative mortality rates in the table")

    if (mtr.dtypes[0] == dtype('int64')
        and mtr.gt(100_000).values.sum() > 0
        and re.match("^[Mm]*", mode) is not None) or \
            (mtr.dtypes[0] == dtype('float64')
             and mtr.gt(1).values.sum() > 0
             and re.match("^[Qq]*", mode) is not None):
        raise ValueError("Values are above the acceptable range.")
