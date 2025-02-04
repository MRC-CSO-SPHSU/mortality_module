from typing import Any

from numpy import dtype, isinf
from pandas import DataFrame
import re

from synthwave.utils.constants import *


def validate_column_names(df: DataFrame, names: list[str, ...]) -> None:
    """Checks if all required columns are present.

    This functions goes through the column names of a DataFrame and checks if
    all the mandatory ones from the corresponding list are there.

    Parameters
    ----------
    df: DataFrame
        A dataframe with basic population data.
    names: list[str, ...]
        A list of column names that must be present.

    Raises
    ------
    ValueError
        When there is a mismatch.
    """

    column_names = df.columns.values.tolist()
    for name in column_names:
        if name not in names:
            raise ValueError(
                f"DataFrame doesn't contain all required columns: {name} is missing"
            )


def validate_column_dtypes(
    df: DataFrame, names: list[str, ...], types: dict[str, Any]
) -> None:
    """Ensures all required columns are of corresponding dtypes.

    It is assumed that all checked columns are present.

    Parameters
    ----------
    df: DataFrame
        A dataframe with basic population data.
    names: list[str, ...]
        Mandatory column names
    types: dict

    Raises
    ------
    ValueError
        When there is a mismatch.
    """

    for name in names:
        if types[name] != df[name].dtype:
            raise ValueError(
                f"Columns are of invalid type: {name} must be "
                f"{types[name]}, but is {df[name].dtype}"
            )


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

    delta = set(df["age_group"].to_list()).difference(set(AGE_GROUPS))
    if len(delta) != 0:
        raise ValueError(
            f"Age groups do not cover the whole range; " f"the difference is {delta}"
        )


def validate_sexes(df: DataFrame) -> None:
    """Ensures only 'm' and 'f' are used as sex values.

    These are the only two supported at the moment.

    Parameters
    ----------
    df: DataFrame
        A dataframe with any population data.

    Raises
    ------
    ValueError
        When there is more than two values.
    """

    delta = set(df["sex"].to_list()).difference(set(SEX_LABELS))
    if len(delta) != 0:
        raise ValueError(
            f"Sexes do not match the expected values; " f"the difference is {delta}"
        )


def validate_size(df: DataFrame) -> None:
    """Makes sure the number of records is divisible by age groups and sexes.

    Parameters
    ----------
    df: DataFrame
        A dataframe with any population data.

    Raises
    ------
    ValueError
        When there is more than two values.
    """

    if df.shape[0] % len(AGE_GROUPS) != 0 | df.shape[0] % len(SEX_LABELS) != 0:
        raise ValueError(
            f"Invalid DataFrame size: {df.shape[0]} is not "
            f"divisible by {len(AGE_GROUPS)} or {len(SEX_LABELS)}"
        )


def validate_mortality_rates(df: DataFrame, mode: str) -> None:
    """Ensures that mortality rates are compliant.

    This function assumes that the mode has been validated already.
    """

    # validate years in range
    # 2011 and later, all consecutive

    # validate age groups im multiindex

    # validate expected dtypes

    if df.isnull().values.any():
        raise ValueError("NaN in the table.")

    if isinf(df).values.sum():
        raise ValueError("Inf in the table.")

    if df.lt(0).values.sum() > 0:
        raise ValueError("Negative mortality rates in the table")

    # validate data values
    # todo this one needs to be reworked
    if (
        df.dtypes[0] == dtype("int64")
        and df.gt(100_000).values.sum() > 0
        and re.match("^[Mm]*", mode) is not None
    ) or (
        df.dtypes[0] == dtype("float64")
        and df.gt(1).values.sum() > 0
        and re.match("^[Qq]*", mode) is not None
    ):
        raise ValueError("Values are above the acceptable range.")
