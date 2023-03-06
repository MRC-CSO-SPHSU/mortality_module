from typing import final

import numpy as np
import pandas as pd
from ctgan import CTGAN

from mortality_module.synthesizer.abstract.base_synthesizer import Synthesizer
from mortality_module.synthesizer.constants import UK_NEW_HH_CODES, \
    CAIND_VALUES, RELFHU_VALUES


class UKCouplesWKids(Synthesizer):
    def __init__(self, seed: int = 13372):
        super().__init__(seed)
        self._hh_class = 'e'

    def run_sanity_checks(self):
        super().run_sanity_checks()

        self._data = self._data[self._data['relhfu'].isin(RELFHU_VALUES.keys())]
        self._data = self._data[self._data['caind'].isin(CAIND_VALUES.keys())]

        bad_ids = self._validate_household_size(self._data)
        if len(bad_ids) > 0:
            print("""Households with inconsistent number of people have been 
                     found, filtering them out.""")
            self._data = self._data[~self._data.index.isin(bad_ids)]

        self._validate_household_composition()

        # parents are of the same sex

        # validate that all hh members are located in the same country;
        # household variables such as country and weight must be identical for all household members

    @staticmethod
    def _validate_household_size(dataset: pd.DataFrame) -> pd.Index:
        """ Validates family structure.

        Any family of this kind must have a father, a mother, and at least one
        child, 3 persons in total.

        Parameters
        ----------
        dataset : pd.DataFrame
            A dataset containing information about households.
        Returns
        -------
        pd.Index
            A list of ids of households that are poorly structured.
        """
        # father, mother and at least one kid;
        # size is at least 3
        print("Validating hh size.")
        t = pd.DataFrame([])
        t['size'] = dataset.groupby('hserialp').count()['age']
        return t[t['size'] < 3].index

    @staticmethod
    def _mean_(data_: pd.DataFrame,
               column_name: str,
               sub_categories: str | list[str, ...]) -> pd.DataFrame:
        data_['mean_' + column_name] = (data_
                                        .groupby(sub_categories,
                                                 group_keys=False)[column_name]
                                        .apply(lambda x: x - x.mean()))
        return data_

    @staticmethod
    def _median_(data_: pd.DataFrame,
                 column_name: str,
                 sub_categories: str | list[str, ...]) -> pd.DataFrame:
        data_['median_' + column_name] = (data_
                                          .groupby(sub_categories,
                                                   group_keys=False)[
                                              column_name]
                                          .apply(lambda x: x - x.median()))
        return data_

    @staticmethod
    def _age_gap_(data_: pd.DataFrame, column_a: str, column_b: str):
        data_['age_gap'] = data_[column_a] - data_[column_b]
        return data_

    @staticmethod
    def _age_gap_kids_(data_: pd.DataFrame,
                       total_children: int) -> pd.DataFrame:
        for i in range(total_children - 1):
            data_[f'kag{i}{i + 1}'] = data_[f'kage{i}'] - data_[f'kage{i + 1}']
        return data_

    @staticmethod
    def _mean_age_kids_hh_(data_: pd.DataFrame,
                           total_children: int) -> pd.DataFrame:
        cl = [f'kage{i}' for i in range(total_children)]
        data_['mean_age_hh'] = data_[cl].mean(axis=1)
        return data_

    @staticmethod
    def _median_age_kids_hh_(data_: pd.DataFrame,
                             total_children: int) -> pd.DataFrame:
        cl = [f'kage{i}' for i in range(total_children)]
        data_['mean_age_hh'] = data_[cl].median(axis=1)
        return data_

    def augment_data(self, indices: pd.Series = None, nkids: int = None):

        t = self._data[self._data.index.isin(indices)]

        augmented_data = pd.DataFrame([], index=indices)

        # father age
        augmented_data['fage'] = t[['relhfu',
                                    'age',
                                    'sex']][(t['relhfu'] != 3) &
                                            (t['sex'] == 'm')]['age']
        # mother age
        augmented_data['mage'] = t[['relhfu',
                                    'age',
                                    'sex']][(t['relhfu'] != 3) &
                                            (t['sex'] == 'f')]['age']
        # hh country
        augmented_data['country'] = t[['relhfu',
                                       'country']][t['relhfu'] == 1]['country']

        # HH weight
        augmented_data['weight'] = t[['relhfu',
                                      'phhwt14']][t['relhfu'] == 1]['phhwt14']

        if nkids == 1:
            kid_info = t[['sex', 'age', 'relhfu']][t['relhfu'] == 3]
            augmented_data['kage0'] = kid_info['age']
            augmented_data['ksex0'] = kid_info['sex']
        else:
            child_ages = (t[['sex',
                             'age',
                             'relhfu']][t['relhfu'] == 3]
                          .groupby(level=0, group_keys=False)
                          .apply(lambda x: x.sort_values(ascending=False,
                                                         by='age'))
                          .drop(columns='relhfu'))
            augmented_data[[f'kage{i}' for i in range(nkids)]] = None
            augmented_data[[f'ksex{i}' for i in range(nkids)]] = None
            for id_ in indices:
                augmented_data.loc[id_, [f'kage{i}' for i in range(nkids)]] = \
                    child_ages.loc[id_]['age'].values
                augmented_data.loc[id_, [f'ksex{i}' for i in range(nkids)]] = \
                    child_ages.loc[id_]['sex'].values

        augmented_data = (augmented_data.
                          pipe(self._mean_, column_name='fage',
                               sub_categories='country').
                          pipe(self._mean_, column_name='mage',
                               sub_categories='country').
                          pipe(self._median_, column_name='fage',
                               sub_categories='country').
                          pipe(self._median_, column_name='mage',
                               sub_categories='country').
                          pipe(self._age_gap_, column_a='fage', column_b='mage')
                          )

        if nkids == 1:
            augmented_data['kage0'] = augmented_data['kage0'].astype(int)
            augmented_data = (augmented_data
                              .pipe(self._mean_, column_name='kage0',
                                    sub_categories='country')
                              .pipe(self._median_, column_name='kage0',
                                    sub_categories='country'))
        else:
            for i in range(nkids):
                augmented_data[f'kage{i}'] = augmented_data[f'kage{i}'].astype(
                    int)

            augmented_data = (augmented_data
                              .pipe(self._age_gap_kids_, total_children=nkids,
                                    sub_categories='country')
                              .pipe(self._mean_age_kids_hh_,
                                    total_children=nkids,
                                    sub_categories='country')
                              .pipe(self._median_age_kids_hh_,
                                    total_children=nkids,
                                    sub_categories='country'))

        return self.scale_sample(augmented_data)

    def scale_sample(self, df_):
        d = pd.DataFrame(
            np.concatenate([np.repeat([df_.loc[id_].values],
                                      df_.loc[id_]['weight'],
                                      axis=0) for id_ in df_.index]),
            columns=df_.columns)

        d['hh_id'] = self.generate_hh_id(len(d.index))

        return d.drop(columns=['weight'])
        # TODO check the description of weights, they must be applied to selected columns only

    def generate_new_population(self) -> pd.DataFrame:
        self.data_preprocessing()
        self.extract_subset(('country',
                             'sex',
                             'age',
                             'phhwt14',
                             'hhtype6',
                             'hserialp',
                             'relhfu',
                             'caind'),
                            tuple(UK_NEW_HH_CODES[self._hh_class]),
                            'hhtype6')

        self._data.set_index('hserialp', inplace=True)
        self.run_sanity_checks()

        # get number of kids per hh
        total_kids = pd.DataFrame([],
                                  index=np.unique(self._data.index),
                                  dtype=int)

        total_kids['total_kids_hh'] = self._data['relhfu'][
            self._data['relhfu'] == 3].groupby(level=0).count().astype(int)

        # split by number of kids
        min_kids = 1
        max_kids = total_kids.max().values[0]

        result = []
        for i in range(min_kids, max_kids + 1):
            temp = self.augment_data(
                indices=total_kids[total_kids['total_kids_hh'] == i].index,
                nkids=i)

            temp = temp.sample(frac=1).reset_index(drop=True).drop(
                columns='hh_id')

            model = self.train_model(temp, [f'ksex{k}' for k in range(i)])

            new_sample = self.sample_model(model, len(temp.index))
            new_sample['hh_id'] = self.generate_hh_id(len(temp.index))

            ns1 = new_sample.set_index('hh_id')
            ns2 = new_sample.melt(id_vars=['country', 'hh_id'],
                                  value_vars=['fage', 'mage', 'kage0'],
                                  var_name='sex',
                                  value_name='age').set_index('hh_id')

            ns2.loc[ns2['sex'] == 'fage', ['sex_']] = 'f'
            ns2.loc[ns2['sex'] == 'mage', ['sex_']] = 'm'

            for j in range(i):
                ns2.loc[ns2['sex'] == f'kage{j}', ['sex_']] = ns1[f'ksex{j}']
                ns2.loc[ns2['sex'] == f'kage{j}', ['child_indicator']] = 1

            ns2['child_indicator'] = ns2['child_indicator'].fillna(0)
            ns2 = ns2.drop(columns=['sex']).rename(columns={"sex_": "sex"})
            ns2['child_indicator'] = ns2['child_indicator'].astype(int)

            result.append(ns2)

        return pd.concat(result)

    @staticmethod
    def train_model(data, extra_discrete_columns: list = None):
        # Names of the columns that are discrete
        print('Start training')
        print([cn for cn in data.columns])
        dc = ['country']
        if extra_discrete_columns is not None:
            dc += extra_discrete_columns

        ctgan = CTGAN(verbose=True,
                      batch_size=150_000  # (int((len(data.index) // 50) / 2))*2
                      )
        ctgan.fit(data, dc)
        print('Done training')
        return ctgan

    @staticmethod
    def sample_model(model_: CTGAN, sample_size: int):
        return model_.sample(sample_size)

    def transform_table(self, table):
        pass

    def _validate_household_composition(self):
        grouped = self._data.groupby(level=0)

        mh = []  # head of the household is there
        mp = []  # partner is there
        mk = []  # there are actually kids in a family with kids

        # todo replace with better code
        for name, group in grouped:
            if 1 not in group['relhfu'].values:
                mh.append(name)
            if 2 not in group['relhfu'].values:
                mp.append(name)
            if 3 not in group['relhfu'].values:
                mk.append(name)

        if (len(mh) > 0) | (len(mp) > 0) | (len(mk) > 0):
            print("Incomplete records detected, filtering them out.")
            self._data.drop(self._data[(self._data.index.isin(mh)) |
                                       (self._data.index.isin(mp)) |
                                       (self._data.index.isin(mk))].index,
                            inplace=True)


if __name__ == "__main__":
    ukcwk = UKCouplesWKids()
    ukcwk.read_data(input())
    ukcwk.generate_new_population().to_csv('/tmp/couples_with_kids.csv',
                                           index=False)
