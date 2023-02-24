import numpy as np
import pandas as pd


class Sanitizer:
    @classmethod
    def household_size(cls, dataset: pd.DataFrame, id_column_name: str,
                       expected_size: int):
        if id_column_name in dataset.columns:
            t = dataset[id_column_name]
            return [id_ for id_ in np.unique(t) if
                    t[t == id_].size != expected_size]
        else:
            print(f"No {id_column_name} column found, check your data.")
            return []

    @classmethod
    def age(cls, data, age_min, age_max):
        # non negative, less than 100?
        pass

    @classmethod
    def sex(cls, data, values):
        # m and f
        pass

    @classmethod
    def country(cls, data, values):
        # uk only
        pass

    @classmethod
    def weights(cls, data):
        # positive, no zeros
        pass

    @classmethod
    def sex_in_couple(cls):
        # only males and females, no same sex
        pass

    @classmethod
    def household_roles(cls):
        # one head, one partner? + kids
        pass

    @classmethod
    def child_adult(cls):
        # Child/Adult indicator caind should make sense
        pass
