from mortality_module.synthesizer.abstract.base_synthesizer import Synthesizer
import pandas as pd
import numpy as np
from scipy import stats
import itertools as it
import math as m

from mortality_module.synthesizer.constants import COUNTRY_MAP, SEX_MAP


class UKSinglePersonHH(Synthesizer):
    def __init__(self, seed=1337):
        super().__init__(seed)

    def run_sanity_checks(self):
        pass

    def extract_subset(self, column_names, hh_codes):
        # TODO codes coule be an int or an iterable of integers
        self._data = self._df[self._df['HHTYPE6'] ==
                              hh_codes][column_names].reset_index(drop=True)
        # NOTE: different individuals of the same age have different weights!

    def augment_data(self):
        pass

    def generate_new_population(self):
        pass

    def read_data(self, file_name):
        self._raw_data = pd.read_spss(file_name, convert_categoricals=False)
        self._df = self._raw_data.copy(deep=True)
        # TODO add options for various data formats

    def data_preprocessing(self):
        self._df['COUNTRY'] = self._df['COUNTRY']\
            .replace(4, 3)\
            .replace(5, 4)\
            .astype(int)\
            .replace(COUNTRY_MAP)
        self._df['SEX'] = self._df['SEX']\
            .astype(int)\
            .replace(SEX_MAP)
        self._df['AGE'] = self._df['AGE'].astype(int)
        # FIXME add control sums everywhere!!!


    def populator(self, parameters, total_data):
        dataset = total_data[(total_data["SEX"] == parameters[0]) &
                             (total_data["COUNTRY"] == parameters[1])]['AGE'].values

        w = total_data[(total_data["SEX"] == parameters[0]) &
                       (total_data["COUNTRY"] == parameters[1])]['PHHWTA14'].values

        age_min = min(dataset)
        age_max = max(dataset)
        num_bins = age_max - age_min + 1

        result = np.histogram(dataset,
                              bins=num_bins,
                              range=[age_min, age_max + 1],
                              weights=w,
                              density=True)
        assert m.fsum(result[0]) == 1, 'Probabilities must add up to 1.'
        return result

    def population_pyramid(self, raw_data, parameters):
        densities, bin_edges = self.populator(parameters, raw_data)
        return stats.rv_discrete(name='population_pyramid',
                                 values=(bin_edges[:-1], densities))

    def populate_single_household(self):
        actual_size = self._data[['SEX', 'COUNTRY', 'PHHWTA14']].\
            groupby(['SEX', 'COUNTRY']).\
            sum(['PHHWTA14'])

        df_collection = [pd.DataFrame()] * 8

        control_sum = 0

        for index_, combination in enumerate(it.product(('m', 'f'),
                                                        ('e', 'w', 's', 'ni'))):
            pp = self.population_pyramid(self._data, combination)

            sample_size = int(actual_size.loc[combination])
            control_sum += sample_size

            d = {'AGE': pp.rvs(size=sample_size),
                 'SEX': combination[0],
                 'COUNTRY': combination[1],
                 'HHID': self.generate_hh_id(sample_size)}

            df_collection[index_] = pd.DataFrame(data=d)

        result = pd.concat(df_collection, ignore_index=True)

        result['AGE'] = result['AGE'].astype(int).astype('category')
        result['SEX'] = result['SEX'].astype('category')
        result['COUNTRY'] = result['COUNTRY'].astype('category')

        result['HH_TYPE'] = 1

        assert int(control_sum) == result['AGE'].size, "Population size mismatch"

        return result

    def cancel_changes(self):
        self._data = None


if __name__ == "__main__":
    uksphh = UKSinglePersonHH()
    uksphh.read_data(input())
    uksphh.data_preprocessing()
    uksphh.extract_subset(['COUNTRY', 'SEX', 'AGE', 'PHHWTA14'], 1)

    uksphh.populate_single_household().to_csv('single_person_household.csv',
                                              index=False)