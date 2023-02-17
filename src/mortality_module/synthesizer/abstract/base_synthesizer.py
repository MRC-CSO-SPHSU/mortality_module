from abc import ABC, abstractmethod
import numpy as np
import random

class Synthesizer(ABC):
    def __init__(self, seed=1337):
        self._rng = random.Random()
        self._rng_seed = seed
        self._update_rng_state()

        self._raw_data = None
        self._df = None

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
    def extract_columns(self, column_names):
        pass

    @abstractmethod
    def augment_data(self):
        pass

    @abstractmethod
    def generate_new_population(self):
        pass