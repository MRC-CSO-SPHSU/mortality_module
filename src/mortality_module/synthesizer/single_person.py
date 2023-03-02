import itertools as it
import math as m

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from mortality_module.synthesizer.abstract.base_synthesizer import Synthesizer
from mortality_module.synthesizer.sanitizer import Sanitizer
from mortality_module.synthesizer.utils import data_range


class UKSinglePersonHH(Synthesizer):
    def __init__(self, seed: int = 1337):
        super().__init__(seed)

    def run_sanity_checks(self):
        super().run_sanity_checks()
        bad_ids = self._validate_household_size(self._data)
        if len(bad_ids) > 0:
            print("""Households with inconsistent number of people have been
             found, filtering them out.""")

    @staticmethod
    def _validate_household_size(dataset):
        """Ensures that every household is composed of one person only."""
        return Sanitizer.household_size(dataset, 'hserialp', 1)

    def augment_data(self):
        raise NotImplementedError("No augmentation is needed.")

    def generate_new_population(self) -> pd.DataFrame:
        self.data_preprocessing()
        self.extract_subset(('country', 'sex', 'age', 'phhwta14', 'hhtype6'), 1,
                            'hhtype6')
        self.run_sanity_checks()
        return self.populate_single_household()

    def build_age_distribution(self,
                               sex_: str,
                               country_: str) -> tuple[np.ndarray, np.ndarray]:
        t = self._data[(self._data["sex"] == sex_) &
                       (self._data["country"] == country_)]

        ages = t['age'].values
        w = t['phhwta14'].values

        num_bins, range_ = data_range(ages)

        result = np.histogram(ages,
                              bins=num_bins,
                              range=range_,
                              weights=w,
                              density=True)
        assert m.fsum(result[0]) == 1, 'Probabilities must add up to 1.'
        return result

    def init_sampler(self, sex_: str, country_: str) -> stats.rv_discrete:
        densities, bin_edges = self.build_age_distribution(sex_, country_)
        return stats.rv_discrete(name='population_pyramid',
                                 values=(bin_edges[:-1], densities))

    def populate_single_household(self) -> pd.DataFrame:
        population_size = self._data[['sex', 'country', 'phhwta14']]. \
            groupby(['sex', 'country']). \
            sum(['phhwta14'])

        df_collection = []

        control_sum = 0

        for (sex_code, country_code) in tqdm(it.product(('m', 'f'),
                                                        ('e', 'w', 's', 'ni'))):
            age_distribution = self.init_sampler(sex_code, country_code)

            sample_size = int(population_size.loc[sex_code, country_code])

            control_sum += sample_size

            df_collection.append(pd.DataFrame(data={
                'age': age_distribution.rvs(size=sample_size),
                'sex': sex_code,
                'country': country_code,
                'hh_id': self.generate_hh_id(sample_size)}))

        result = pd.concat(df_collection, ignore_index=True)

        result['age'] = result['age'].astype(int)
        result['sex'] = result['sex']
        result['country'] = result['country']

        result['hh_type'] = 1

        assert int(control_sum) == result['age'].size, "Size mismatch"

        return result


if __name__ == "__main__":
    uksphh = UKSinglePersonHH()
    uksphh.read_data(input())
    uksphh.generate_new_population().to_csv('single_person_household.csv',
                                            index=False)
