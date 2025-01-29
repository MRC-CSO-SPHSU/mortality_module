from pathlib import Path

import pandas as pd
from yaml import safe_load

pd.set_option('future.no_silent_downcasting', True)

ETHNIC_MAP = safe_load(Path(
    "../../data/alignment/uk/ethnic_map.yaml").read_text())
TENURE_MAP = safe_load(Path(
    "../../data/alignment/uk/tenure_map.yaml").read_text())
LARPUK_MAP = safe_load(Path(
    "../../data/alignment/uk/larpuk_map.yaml").read_text())
HOUSEHOLD_MAP = safe_load(Path(
    "../../data/alignment/uk/household_map.yaml").read_text())
NAME_MAP = safe_load(Path("../../data/alignment/uk/name_map.yaml").read_text())
REGION_MAP = safe_load(Path(
    "../../data/alignment/uk/region_map.yaml").read_text())

BOOL_MAP = {1: True, 2: False}

# see https://iserredex.essex.ac.uk/support/issues/551

# NOTE try not to alter USoc variables, they are used in labsim
# NOTE pensionable age is 65+

# TODO at this stage we need to follow the definition of family https://www.nisra.gov.uk/sites/nisra.gov.uk/files/publications/2011-census-definitions-and-output-classifications.pdf
#  we only consider cases of 1 family per household

class Harmonizer:
    @staticmethod
    def preprocess_data(_df: pd.DataFrame) -> pd.DataFrame:
        _df.columns = _df.columns.str.lower()
        return _df

    @staticmethod
    def _prune(_df: pd.DataFrame) -> pd.DataFrame:
        _df = _df[NAME_MAP.keys()]
        _df = _df.replace({"region": REGION_MAP}).rename(columns={_k: _v for _k, _v in NAME_MAP.items() if _v is not None})

        _df["indicator_person_sex"] = _df["indicator_person_sex"].replace(BOOL_MAP).astype("bool[pyarrow]")

        _df["hours_person_week_coarse"] = _df["hours_person_week_coarse"].replace({-9: pd.NA}).astype("uint8[pyarrow]")
        # those folks who don't work have zero hours a week; it also might be the case of invalid record

        return _df

    @staticmethod
    def get_ew(_df: pd.DataFrame) -> pd.DataFrame:
        # ENGLAND & WALES

        _df = Harmonizer._prune(_df).replace({'category_person_ethnic_group': {14: 15, 15: 14, 18: 97},
                                              # map to ethn_dv; some categories are not the same
                                              "ordinal_highest_qualification": {-9: 1, 10: 1, 11: 2, 12: 2, 13: 3, 14: 3, 16: 3, 15: 4},
                                              "category_nssec": {3: 3.1, 4: 3.2, 5: 3.3, 6: 3.4,
                                                                 7: 4.1, 8: 4.2, 9: 4.3, 10: 4.4,
                                                                 11: 5,
                                                                 12: 6,
                                                                 13: 7.1, 14: 7.2, 15: 7.3, 16: 7.4,
                                                                 17: 8.1, 18: 8.2,
                                                                 19: 9.1, 20: 9.2,
                                                                 21: 10,
                                                                 22: 11.1, 23: 11.2,
                                                                 24: 12.1, 25: 12.2, 26: 12.3, 27: 12.4, 28: 12.5, 29: 12.6, 30: 12.7,
                                                                 31: 13.1, 32: 13.2, 33: 13.3, 34: 13.4, 35: 13.5,
                                                                 36: 14.1, 37: 14.2,
                                                                 38: 15,
                                                                 39: 16,
                                                                 40: 17
                                                                 },
                                              })


        _df = _df[_df["category_person_ethnic_group"].ne(3)]
        # Unfortunately, we have to remove the whole category; there is no data about such people in Wave 1
        # this is a very small group, and we assume in every household people belong to the same group

        _df = _df[_df["category_household_type"].ne(3)]
        # by doing so we exclude all people from said households, regardless of their actual attributes. their number is very small; the data is difficult to process too.

        # stick to households only
        return  _df[_df["residtype"].eq(2) & _df["popbasesec"].eq(1)].drop(columns=["residtype", "popbasesec"])
        # some short term residents + students, not clear how to treat them

    @staticmethod
    def get_s(_df: pd.DataFrame) -> pd.DataFrame:
        # SCOTLAND
        # TODO write a better method for renaming columns
        _df = (_df.
               rename(columns={"ecopuk11": "ecopuk",
                               "ethhuk11": "ethnicityew",
                               "fmsps11": "fmspuk11",
                               "hrswrkd": "hours",
                               "termind": "popbasesec",
                               "residence_type": "residtype",
                               "sizhuk11": "sizhuk",
                               "tenhuk11": "tenure",
                               "larpuk11": "larpuk",
                               # an extra column for additional analysis
                               "relps11": "religionew",
                               "relhrppuk11": "relato",
                               "hlqps11": "hlqupuk11",
                               "cenheathuk11": "cenheat",
                               "carsno": "carsnoc",
                               "occ": "socmin",
                               "industry": "indgpuk11"
                               }).
               replace({"ahchuk11": HOUSEHOLD_MAP,
                        # scotland merges no adults and 1 adult + children together, the error is negligible
                        "ethnicityew": ETHNIC_MAP,
                        "tenure": TENURE_MAP,
                        "larpuk": LARPUK_MAP,
                        "religionew": {1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9},
                        "ordinal_highest_qualification": {-9: 1, 20: 1, 21: 2, 22: 3, 23: 4, 24: 4}
                        }
                       )
               )

        _df["region"] = 12
        # This way all labels across datasets are identical

        _df = Harmonizer._prune(_df)

        _df = _df[_df["category_person_ethnic_group"].ne(3)]
        # Unfortunately, we have to remove the whole category; there is no data about such people in Wave 1
        # this is a very small group, and we assume in every household people belong to the same group
        # stick to households only

        return _df[_df["residtype"].eq(2) & _df["popbasesec"].ne(2)].drop(columns=["residtype", "popbasesec"])  # our best guess

    @staticmethod
    def get_ni(_df: pd.DataFrame) -> pd.DataFrame:
        # NORTHERN IRELAND

        _df = (_df.
               rename(columns={"ageh": "age",
                               "ecopuk11": "ecopuk",
                               "ethnicityni_g": "ethnicityew",
                               "residence_type": "residtype",
                               "sizhuk11": "sizhuk",
                               "larpuk11": "larpuk",
                               # an extra column for additional analysis
                               "religionni": "religionew"
                               }).
               replace({"ethnicityew": {1: 2, 2: 9},
                        # 1 to 2, 2 to 9; this is the best guess. following scotland the largest non-white majority will be indian
                        "religionew": {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 8, 7: 1, 8: 9},
                        "larpuk": LARPUK_MAP,
                        "ordinal_highest_qualification": {-9: 1, 10: 1, 11: 2, 12: 2, 13: 3, 14: 3, 16: 3, 15: 4}
                        }))

        _df = Harmonizer._prune(_df)


        _df = _df[_df["category_household_type"].ne(3)]
        # by doing so we exclude all people from said households, regardless of their actual attributes. their number is very small; the data is difficult to process too.

        # stick to households only
        return _df[_df["residtype"].eq(2) & _df["popbasesec"].eq(1)].drop(columns=["residtype", "popbasesec"])


def map_qualification(_data):
    if any(map(pd.isna, [_data["category_person_full_highest_qualification"]] + [_data[f"indicator_person_vocational_qualification_{i}"] for i in [j for j in range(1, 16)]])):
        return pd.NA

    a_levels = any([_data["indicator_person_vocational_qualification_5"],
                    _data["indicator_person_vocational_qualification_10"],
                    _data["indicator_person_vocational_qualification_12"],
                    _data["indicator_person_vocational_qualification_13"],
                    _data["indicator_person_vocational_qualification_14"]])

    if _data["category_person_full_highest_qualification"] in [1, 2]:
        return 1

    if _data["category_person_full_highest_qualification"] in [3, 4, 5]:
        return 2

    if _data["category_person_full_highest_qualification"] in [6, 7, 8, 9, 10, 11]:
        return 2 if _data["indicator_person_vocational_qualification_11"] else 3
        # as A-levels they get dominated by vq_11 which is mapped one level higher

    if _data["category_person_full_highest_qualification"] in [12, 14]:
        # 12 and 14 are GCSE etc., which are in the same fashion dominated by A-levels and above
        if _data["indicator_person_vocational_qualification_11"]: # 2 Other higher degree
            return 2
        if a_levels: # A-level etc
            return 3
        return 4

    if _data["category_person_full_highest_qualification"] in [13, 15]:
        if _data["indicator_person_vocational_qualification_11"]: # 2 Other higher degree
            return 2
        if a_levels: # A-level etc
            return 3
        if _data["indicator_person_vocational_qualification_8"] or _data["indicator_person_vocational_qualification_9"]:
            return 4
        return 5

    if _data["category_person_full_highest_qualification"] in [96]:
        if all([not _data[f"indicator_person_vocational_qualification_{i}"] for i in [j for j in range(1, 16)]]): # all vq w/o 96 are False;
            return 9
        if _data["indicator_person_vocational_qualification_11"]: # 2 Other higher degree
            return 2
        if a_levels: # A-level etc
            return 3
        if _data["indicator_person_vocational_qualification_8"] or _data["indicator_person_vocational_qualification_9"]:
            return 4
        return 5
# category_person_coarse_highest_qualification doesn't always follow this logic, some errors perhaps

def get_coarse_mapping(_x):
    match _x:
        case 1 | 2:
            return 4
        case 3:
            return 3
        case 4 | 5:
            return 2
        case 9:
            return 1
