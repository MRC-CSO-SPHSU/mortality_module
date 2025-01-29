from typing import Tuple

import pandas as pd

# 10_000_000_000 is the limit for 1/2 adults households i.e. no more than that in total
# thus hh ids for 1 adult type will range from 1_000_000_000_000 to 1_999_999_999_999 at most; the increment is 100 to allow some buffer for personal ids within the household itself
# it is unlikely though we'll have any hh of such size, but it's a nice pattern to have
# first digit is the total number of adults per hh
# second digit is the mini batch id 0-9
# two digits after that is the microbatch id
# numbers are experimental

def assign_household_id(df_: pd.DataFrame,
                        mini_batch_id_: int = None,
                        micro_batch_id_: int = None,
                        household_size: int = None,
                        region_id: int = None,
                        total_adults: Tuple[int, ...] = (1, 2)) -> pd.DataFrame:
    assert len(df_) <= 10_000_000_000

    if household_size not in total_adults:  # total adults
        raise ValueError(f"Household size not supported yet: {household_size}")

    if not (0 <= mini_batch_id_ <= 9):
        raise ValueError(f"Mini batch id is out of limits: {mini_batch_id_}")

    if not (0 <= micro_batch_id_ <= 99):
        raise ValueError(f"Micro batch id is out of limits: {micro_batch_id_}")

    if not (1 <= region_id <= 99):
        raise ValueError(f"Region id is out of limits: {region_id}")

    def _build_id_base():
        return (household_size * 1_000_000_000_000 +
                mini_batch_id_ * 100_000_000_000 +
                micro_batch_id_ * 1_000_000_000)

    df_['id_household'] = pd.Series(range(_build_id_base(),
                                          _build_id_base() + len(df_) * 100, 100)) + region_id * 10_000_000_000_000
    return df_
