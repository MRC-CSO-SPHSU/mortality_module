import itertools as it
import math as m

import numpy as np
import pandas as pd
from tqdm import tqdm

from mortality_module.synthesizer.abstract.base_synthesizer import Synthesizer
from mortality_module.synthesizer.constants import UK_NEW_HH_CODES
from mortality_module.synthesizer.sanitizer import Sanitizer
from mortality_module.synthesizer.utils import data_range


class UKCouplesHH(Synthesizer):
    def __init__(self, seed: int = 13371):
        super().__init__(seed)
        self._hh_class = 'c'

    def run_sanity_checks(self) -> None:
        super().run_sanity_checks()

        bad_ids = self._validate_household_size(self._data)
        if len(bad_ids) > 0:
            print("""Households with inconsistent number of people have been 
                     found, filtering them out.""")
            self._data = self._data[~self._data['hserialp'].isin(bad_ids)]

        t = self._data.groupby('hserialp')['sex'].value_counts().unstack()
        same_sex = t.loc[(t['f'] == 2) | (t['m'] == 2)].index
        if len(same_sex) > 0:
            self._data.drop(index=same_sex, inplace=True)


    def augment_data(self) -> None:
        self._data = pd.pivot_table(self._data,
                                    values=['age', 'phhwt14'],
                                    index=['hserialp', 'country', 'hhtype6'],
                                    columns=['sex']). \
            reset_index(). \
            drop(columns=('phhwt14', 'f'))

        new_columns = [s1 if s2 == '' else s1 + '_' + str(s2) for (s1, s2) in
                       self._data.columns.tolist()]
        self._data.columns = new_columns

        self._data['hh_w'] = self._data['phhwt14_m']
        self._data.drop(columns='phhwt14_m', inplace=True)
        self._data.rename(columns={"age_f": "f", "age_m": "m", "hh_w": "w"},
                          inplace=True)

    def generate_new_population(self) -> pd.DataFrame:
        self.data_preprocessing()
        self.extract_subset(('country', 'sex', 'age', 'phhwt14', 'hserialp',
                             'hhtype6'),
                            tuple(UK_NEW_HH_CODES[self._hh_class]),
                            'hhtype6')
        self.run_sanity_checks()
        self.augment_data()
        return self.populate_couples()

    @staticmethod
    def _validate_household_size(dataset):
        """Ensures that every household is composed of exactly two people."""
        return Sanitizer.household_size(dataset, 'hserialp', 2)

    def populate_couples(self) -> pd.DataFrame:
        all_data : list = []

        for (country_, hh_type) in tqdm(it.product(('e', 'w', 's', 'ni'),
                                                   (3, 4))):
            t = self._data[(self._data['country'] == country_) &
                           (self._data['hhtype6'] == hh_type)]

            num_bins_f, range_f = data_range(t['f'])
            num_bins_m, range_m = data_range(t['m'])

            dist, ages_f, ages_m = np.histogram2d(t['f'],
                                                  t['m'],
                                                  bins=[num_bins_f, num_bins_m],
                                                  range=[range_f, range_m],
                                                  weights=t['w'],
                                                  density=True)

            assert m.fsum(dist.flatten()) == 1, \
                'Probabilities must add up to 1.'

            total_sample_households = int(t['w'].sum())

            linear_dist = dist.flatten()
            sample_index = np.random.choice(a=linear_dist.size,
                                            p=linear_dist,
                                            size=total_sample_households)
            index_ = np.unravel_index(sample_index, dist.shape)

            ids = self.generate_hh_id(total_sample_households)

            all_data.append(pd.DataFrame(data={'f': ages_f[index_[0]],
                                               'm': ages_m[index_[1]],
                                               'country': country_,
                                               'hh_id': ids}))


        result = pd.concat(all_data,
                           ignore_index=True).melt(
            id_vars=['country', 'hh_id'], value_vars=['f', 'm'],
            var_name='sex', value_name='age')
        result['age'] = result['age'].astype(int)

        result['hh_type'] = self._hh_class
        return result

if __name__ == "__main__":
    ukchh = UKCouplesHH()
    ukchh.read_data(input())
    ukchh.generate_new_population().to_csv('couples.csv', index=False)
