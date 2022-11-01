from typing import Final, List, Annotated

import pandas as pd
import numpy as np

"""Current sex labels, only males and females are supported at the moment."""
SEX_LABELS: Final[List[str, str]] = ["F", "M"]

"""Current age groups, one for each year and a cumulative one for people aged 
100 and older."""
AGE_GROUPS: Final[Annotated[List[str], 101]] = \
    [str(i) for i in range(100)] + ["100 and over"]

"""Main column names used for mortality predictions."""
COLUMN_NAMES: Final = ['Age group', 'Sex', 'Number of people']

"""Data types that correspond to mandatory column names"""
DTYPES: Final = {COLUMN_NAMES[0]: pd.CategoricalDtype,
                 COLUMN_NAMES[1]: pd.CategoricalDtype,
                 COLUMN_NAMES[2]: np.uint32}
