from typing import Final, List, Annotated

import pandas as pd
import numpy as np

"""Current sex labels, only males and females are supported at the moment."""
SEX_LABELS: Final[Annotated[List[str], 2]] = ["f", "m"]

"""Current age groups, one for each year and a cumulative one for people aged 
100 and older."""
AGE_GROUPS: Final[Annotated[List[str], 101]] = [str(i) for i in range(100)] + [
    "100 and over"
]

"""Main column names used for mortality predictions."""
CENSUS_COLUMN_NAMES: Final = ["year", "sex", "age_group", "people_total"]

"""Data types that correspond to mandatory column names"""
CENSUS_DTYPES: Final = {
    CENSUS_COLUMN_NAMES[0]: np.uint16,
    CENSUS_COLUMN_NAMES[1]: pd.CategoricalDtype,
    CENSUS_COLUMN_NAMES[2]: pd.CategoricalDtype,
    CENSUS_COLUMN_NAMES[3]: np.uint32,
}

MORTALITY_COLUMN_NAMES: Final = ["year", "sex", "age_group", "mortality_rate"]

MORTALITY_DTYPES: Final = {
    CENSUS_COLUMN_NAMES[0]: np.uint16,
    CENSUS_COLUMN_NAMES[1]: pd.CategoricalDtype,
    CENSUS_COLUMN_NAMES[2]: pd.CategoricalDtype,
    CENSUS_COLUMN_NAMES[3]: np.float64,
}
