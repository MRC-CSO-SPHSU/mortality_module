import typing

from numpy import uint32, uint16, ceil
from pandas import DataFrame, MultiIndex
from itertools import product
import mortality_module
from mortality_module.utils.constants import AGE_GROUPS, SEX_LABELS


class Ageing:
    def __init__(
        self,
        population_pyramid: DataFrame,
        mortality_rates: DataFrame,
        birth_numbers: DataFrame,
        end_year: uint16 = 2020,
    ):

        self.pop = population_pyramid
        self.mortality = mortality_rates
        self.birth_numbers = birth_numbers
        self.end_year = end_year

        self.index = [
            (a, s, y)
            for a, s, y in product(range(2011, (end_year + 2)), SEX_LABELS, AGE_GROUPS)
        ]
        mi = MultiIndex.from_tuples(self.index, names=("year", "sex", "age_group"))
        columns = ["people_total", "mortality_rate", "dead_within_year"]
        self.df = DataFrame(index=mi, columns=columns)

        self._validate_population_pyramid()
        self._validate_mortality_rates()
        self._validate_birth_numbers()
        self._validate_end_year()

    def _validate_end_year(self) -> None:
        # 2011 and last year in mort table inclusive
        pass

    def _validate_population_pyramid(self) -> None:
        pass

    def _validate_mortality_rates(self) -> None:
        pass

    def _validate_birth_numbers(self) -> None:
        pass

    def _increment_age(self, df: DataFrame) -> DataFrame:
        pass

    def _add_new_people(self, df: DataFrame) -> DataFrame:
        pass

    @staticmethod
    def _calculate_dead_people(
        df: DataFrame, mortality: DataFrame, year: str = "2011"
    ) -> DataFrame:
        dead = df.copy().drop("Number of people")
        dead["Dead"] = 0.0

        for s in SEX_LABELS:
            for i in range(100):
                age_value = str(i)
                mr = mortality.loc[
                    (mortality["year"] == year) & (mortality["sex"] == s), age_value
                ]
                dead.loc[(dead["Age"] == age_value) & (dead["sex"] == s), age_value] = (
                    mr
                    * df.loc[(dead["Age"] == age_value) & (dead["sex"] == s), age_value]
                )

        # treat 100 and over separately somehow?
        # mortality tables: rows are years, columns are ages that year + sex

        return dead

    def _remove_dead_people(self):
        pass

    def run(self) -> DataFrame:
        # populate year 2011
        self.df["people_total"] = self.pop["people_total"]

        # add all mortality rates for all years
        self.df["mortality_rate"] = self.mortality["mortality_rate"]

        for year in range(2011, (self.end_year + 1)):
            # calculate the number of dead people
            # do we round up or down?
            dead = self.df.loc[year, ["people_total"]]
            dead *= self.df.loc[year, ["mortality_rate"]].values

            dead /= 100_000
            dead = dead.apply(lambda row: ceil(row)).astype("uint32")
            self.df.loc[year, "dead_within_year"] = dead.values

            # calculate pop - deaths next year and age it
            # at the moment all people who reach 100 die immediately.
            pop_next_year = self.df.loc[year, ["people_total"]]
            pop_next_year -= self.df.loc[year, ["dead_within_year"]].values
            pop_next_year = pop_next_year.groupby(level=0).shift(1)

            births_next_year = self.birth_numbers.loc[year]
            births_next_year.rename(
                columns={"birth_number": "people_total"}, inplace=True
            )

            pop_next_year.update(births_next_year)

            self.df.loc[[year + 1], "people_total"] = pop_next_year[
                "people_total"
            ].values

        return self.df


def ageing(df: DataFrame, birth_number: typing.Dict[str, uint32]) -> DataFrame:
    """Ages every age group by one year.

    Effectively shifts the data by 1 year, replacing the first row with the
    number of newborns. In addition, the last age group that is `100 and over`
    is a cumulative variable and needs to be updated separately.

    :param df:
    :param birth_number:
    :return:
    """

    # validate_column_names(df)
    # validate_column_dtypes(df)
    # validate_age_groups(df)
    # validate_sexes(df)
    # validate_size(df)

    # todo validate birth number

    for s_value in SEX_LABELS:
        old_elderly_value = df.loc[
            df["Age group"] == "100 and over", df["Sex"] == s_value, "Number of people"
        ]

        df.loc[df["Sex"] == s_value] = df.loc[df["Sex"] == s_value].shift(
            1, fill_value=birth_number[s_value]
        )

        df.loc[
            df["Age group"] == "100 and over", df["Sex"] == s_value, "Number of people"
        ] += old_elderly_value

    return df


if __name__ == "__main__":
    cen = mortality_module.load_census_data()
    mo = mortality_module.load_mortality_rates()
    b = mortality_module.load_birth_numbers()
    a = Ageing(cen, mo, b)
    data = a.run()
    data.to_csv("result.csv")

    import numpy as np
    import matplotlib.pylab as pl

    pl.figure()
    colors = pl.cm.jet(np.linspace(0, 1, 11))

    for i, year in enumerate(range(2011, (2020 + 1))):
        data.loc[(year, "f", slice(None))]["people_total"].plot(
            color=colors[i], figsize=(32, 24), fontsize=26, label=str(year)
        )

    pl.grid(axis="both", color="0.95")
    pl.legend(title="Year of study:")
    pl.title("Number of people per age group")
    pl.savefig("result.png")
    # pl.show()
