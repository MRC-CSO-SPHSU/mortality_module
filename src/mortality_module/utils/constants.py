from typing import Final, List, Annotated

import pandas as pd
import numpy as np


"""Encoding for constituent countries"""
COUNTRY_CODES: Final[Annotated[List[str], 4]] = ["e", "w", "s", "ni"]


"""Current sex labels, only males and females are supported at the moment."""
SEX_LABELS: Final[Annotated[List[str], 2]] = ["f", "m"]

"""Current age groups, one for each year and a cumulative one for people aged 
100 and older."""
AGE_GROUPS: Final[Annotated[List[str], 101]] = [str(i) for i in range(100)] + [
    "100 and over"
]

"""Main column names used for mortality predictions."""
CENSUS_COLUMN_NAMES: Final[Annotated[List[str], 5]] = [
    "year",
    "sex",
    "age_group",
    "country",
    "people_total",
]

"""Data types that correspond to mandatory column names"""
CENSUS_DTYPES: Final = {
    CENSUS_COLUMN_NAMES[0]: np.uint16,
    CENSUS_COLUMN_NAMES[1]: pd.CategoricalDtype,
    CENSUS_COLUMN_NAMES[2]: pd.CategoricalDtype,
    CENSUS_COLUMN_NAMES[3]: pd.CategoricalDtype,
    CENSUS_COLUMN_NAMES[4]: np.uint32,
}

MORTALITY_COLUMN_NAMES: Final[Annotated[List[str], 5]] = [
    "year",
    "sex",
    "age_group",
    "country",
    "mortality_rate",
]

MORTALITY_DTYPES: Final = {
    CENSUS_COLUMN_NAMES[0]: np.uint16,
    CENSUS_COLUMN_NAMES[1]: pd.CategoricalDtype,
    CENSUS_COLUMN_NAMES[2]: pd.CategoricalDtype,
    CENSUS_COLUMN_NAMES[3]: pd.CategoricalDtype,
    CENSUS_COLUMN_NAMES[4]: np.float64,
}
