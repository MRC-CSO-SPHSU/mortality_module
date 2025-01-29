import random
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import final, Final
import os
import numpy as np
import pandas as pd
import torch

from mortality_module.utils.general import path_validation

# no country link, no particular dataset link, universal methods

class Synthesizer(ABC):
    def __init__(self, seed: int = 1337, deterministic: bool = True):
        self._rng: Final = random.Random()
        self._rng_seed: int = seed

        self._raw_individuals = None
        self._individuals = None

        self._raw_households = None
        self._households = None

        self.transformers = dict()

        if deterministic:
            print('Using deterministic models; that might take more time, '
                  'however, all results should be reproducible.')

            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(deterministic)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        self._update_rng_state()
        self.degenerate_distribution: dict = {}

    @final
    def _find_degenerate_fields(self, c_: str):
        # must be used when all values are converted to np.nan, pd.NA, None
        if len(x := pd.unique(self._individuals[c_].fillna(pd.NA))) == 1:
            if pd.isna(v := x[0]):
                print('Invalid column, contains only nans.')
            self.degenerate_distribution[c_] = {'value': v, 'type': type(v)}
            self._individuals.drop(columns=c_)
        if len(x) == 2:
            if (s := pd.Series(x)).isna().any():
                v = s.to_list()[0]
                self.degenerate_distribution[c_] = {'value': v,
                                                    'type': type(v)}
                self._individuals.drop(columns=c_)

    @final
    def _update_rng_state(self) -> None:
        """Updates all used RNGs to employ the same (new) seed.
        """
        self._rng.seed(self._rng_seed)
        random.seed(self._rng_seed)
        np.random.seed(self._rng_seed)
        torch.manual_seed(self._rng_seed)

    @final
    @property
    def rng_seed(self) -> int:
        return self._rng_seed

    @property
    def data(self, individuals=True) -> pd.DataFrame:
        """Returns a deep copy of the working dataset.
        """
        if individuals:
            return self._individuals.copy()
        else:
            return self._households.copy()

    @final
    @rng_seed.setter
    def rng_seed(self, value: int) -> None:
        """Updates the seed and resets the state of every RNG.
        """
        self._rng_seed = value
        self._update_rng_state()

    def read_data(self, file_name: str | Path, individuals=True) -> None:
        """Reads a .pkl file with individual/household data.

        Parameters
        ----------
        file_name : str or Path
            The full file path.

        """
        p_ = path_validation(file_name)
        if individuals:
            self._raw_individuals = pd.read_pickle(p_)
            self._individuals = self._raw_individuals.copy()
        else:
            self._raw_households = pd.read_pickle(p_)
            self._households = self._raw_households.copy()

    def run_sanity_checks(self) -> None:
        # TODO assert there is no NA, nan, None, OPCODES
        pass

    def extract_subset(self, household_size: int | None) -> None:
        matched_hh = (self
                      ._individuals
                      .groupby('id_household')
                      .size() == household_size).to_frame().rename(
            columns={0: 'match'})
        matched_hh = matched_hh[matched_hh['match']].index
        all_indices = self._individuals['id_household']
        self._individuals = self._individuals[
            all_indices.isin(matched_hh)]

        self._individuals = self._individuals.merge(self._households,
                                                    how='inner',
                                                    left_on=['id_household'],
                                                    right_on=['id_household'])
        print('Dropping zero weights')
        self._individuals = self._individuals[self._individuals['weight_household'] != 0]

    @abstractmethod
    def generate_new_population(self):
        pass

    @final
    def generate_hh_id(self, ss: int) -> list[uuid.UUID, ...]:
        # FIXME we use a different method
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
        self._individuals = self._raw_individuals.copy()
        self._households = self._raw_households.copy()

    @staticmethod
    @abstractmethod
    def transform_table(*args, **kwargs):
        pass

    def drop_ids(self):
        """Removes id columns from the table with individuals"""
        _to_drop = []
        for _c in self._individuals.columns:
            if _c.startswith("id_"):
                _to_drop.append(_c)
        self._individuals = self._individuals.drop(columns=_to_drop)
