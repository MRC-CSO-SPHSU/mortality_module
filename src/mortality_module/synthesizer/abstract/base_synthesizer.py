import random
import uuid
from abc import ABC, abstractmethod
from typing import Tuple, final

import numpy as np
import pandas as pd

from mortality_module.synthesizer.constants import UK_COUNTRY_MAP, UK_SEX_MAP


class Synthesizer(ABC):
    def __init__(self, seed: int = 1337):
        self._rng = random.Random()
        self._rng_seed = seed

        self._update_rng_state()
        # ensure the same data generated every time we run this from scratch

        self._raw_data = None
        self._data = None

        self._hh_class = None

    @final
    def _update_rng_state(self):
        self._rng.seed(self._rng_seed)
        np.random.seed(seed=self._rng_seed)

    @final
    @property
    def rng_seed(self):
        return self._rng_seed

    @final
    @rng_seed.setter
    def rng_seed(self, value):
        self._rng_seed = value
        self._update_rng_state()

    def read_data(self, file_name: str) -> None:
        """Reads a binary SPSS file and converts it to a DataFrame

        Parameters
        ----------
        file_name : str
            The name of a file.
        """
        self._raw_data = pd.read_spss(file_name, convert_categoricals=False)
        self._data = self._raw_data.copy(deep=True)
        self._data.columns = self._data.columns.str.lower()


    def run_sanity_checks(self) -> None:
        """Runs basic data checks.

        Makes sure that all countries, sex values, and age are in some
        acceptable range. Also asserts that hh weights are strictly positive.
        """
        print("Checking country, sex, age, and weights.")
        assert set(self._data['country']).issubset(UK_COUNTRY_MAP.values())
        assert set(self._data['sex']).issubset(UK_SEX_MAP.values())
        assert 0 <= self._data['age'].min() <= 100

        n = 'phhwta14' if 'phhwta14' in self._data.columns else 'phhwt14'
        self._data = self._data[self._data[n] > 0]
        assert self._data[n].min() > 0


    @final
    def extract_subset(self,
                       column_names: Tuple[str, ...],
                       hh_codes: Tuple[int, ...],
                       household_column_name: str) -> None:
        """Selects a subset of data.

        Selects certain properties of a household, including its actual type.

        Parameters
        ----------
        column_names : Tuple[str, ...]
            Names of the columns to be selected.
        hh_codes : Tuple[int, ...]
            A collection of integers that encode households of some types.
        household_column_name : str
            The column that contains household ids.

        Notes
        -----
            different households with the same parameters might have different
            weights.
        """

        hh_match = self._data[household_column_name] == hh_codes[0]
        for val in hh_codes[1:]:
            hh_match = hh_match | (self._data[household_column_name] == val)

        self._data = (self._data[hh_match][list(column_names)]
                      .reset_index(drop=True))

    @abstractmethod
    def augment_data(self, *args, **kwargs):
        pass

    @abstractmethod
    def generate_new_population(self):
        pass

    @final
    def data_preprocessing(self):
        self._data['country'] = (self._data['country']
                                 .replace(4, 3)
                                 .replace(5, 4)
                                 .astype(int)
                                 .replace(UK_COUNTRY_MAP))
        self._data['sex'] = (self._data['sex']
                             .astype(int)
                             .replace(UK_SEX_MAP))
        self._data['age'] = self._data['age'].astype(int)
        if 'hserialp' in self._data.columns:
            self._data['hserialp'] = self._data['hserialp'].astype(int)

    def generate_hh_id(self, ss: int) -> list[uuid.UUID, ...]:
        """Generates unique household ids.

        Parameters
        ----------
        ss : int
            Sample size.
        Returns
        -------
        list
            A list of UUID objects of the given size.
        """
        return [uuid.UUID(int=self._rng.getrandbits(128)) for _ in range(ss)]

    @final
    def cancel_changes(self):
        self._data = None

    @staticmethod
    def _validate_household_size(dataset):
        pass
