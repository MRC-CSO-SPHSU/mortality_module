import uuid
from abc import ABC, abstractmethod
import numpy as np
import random

class Synthesizer(ABC):
    def __init__(self, seed=1337):
        self._rng = random.Random()
        self._rng_seed = seed

        self._update_rng_state() # ensure the same data generated every time we run this from scratch

        self._raw_data = None
        self._df = None
        self._data = None

    def _update_rng_state(self):
        self._rng.seed(self._rng_seed)
        np.random.seed(seed=self._rng_seed)

    @property
    def rng_seed(self):
        return self._rng_seed

    @rng_seed.setter
    def rng_seed(self, value):
        self._rng_seed = value
        self._update_rng_state()

    @abstractmethod
    def read_data(self, file_name):
        pass

    @abstractmethod
    def run_sanity_checks(self):
        # run sanitizer
        pass

    @abstractmethod
    def extract_subset(self, column_names, hh_codes):
        pass

    @abstractmethod
    def augment_data(self):
        pass

    @abstractmethod
    def generate_new_population(self):
        pass

    @abstractmethod
    def data_preprocessing(self):
        pass

    def generate_hh_id(self, ss: int) -> list:
        return [uuid.UUID(int=self._rng.getrandbits(128)) for _ in range(ss)]

    @staticmethod
    def _age_range(data):
        age_min = min(data)
        age_max = max(data)

        num_bins = age_max - age_min + 1
        range_ = [age_min, age_max + 1]

        return num_bins, range_