from pandas import DataFrame, read_csv
from importlib import resources


def load_census_data() -> DataFrame:
    """Returns a dataframe with the 2011 census data.

    Uses the following indices:
        year            1 non-null uint16;

        sex             2 non-null object;

        age_group       101 non-null object;

    Contains the following field:
        people_total    202 non-null uint32;
    """

    with resources.path("mortality_module.data", "uk_2011_census.csv") as f:
        df = read_csv(f).set_index(["year", "sex", "age_group"])
        df.people_total = df.people_total.astype("uint32")
    return df.astype("uint32")


def load_mortality_rates() -> DataFrame:
    """Returns a dataframe with the UK mortality rates, actual and projections.

    Uses the following indices:
        year            245 non-null uint16;

        sex             2 non-null object;

        age_group       101 non-null object;

    Contains the following field:
        mortality_rate    49490 non-null float64;
    """

    with resources.path("mortality_module.data", "uk_2020_mortality.csv") as f:
        df = read_csv(f).set_index(["year", "sex", "age_group"])
    return df


def load_birth_numbers() -> DataFrame:
    """Returns a dataframe with the UK birth numbers, actual values only.

    Uses the following indices:
        year            11 non-null uint16;

        sex             2 non-null object;

        age_group       1 non-null object;

    Contains the following field:
        birth_number    22 non-null uint32;
    """

    with resources.path("mortality_module.data", "uk_2020_birth.csv") as f:
        df = read_csv(f)

    df["age_group"] = df["age_group"].astype(str)
    return df.set_index(["year", "sex", "age_group"]).astype("uint32")
