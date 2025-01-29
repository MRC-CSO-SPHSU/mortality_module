import pandas as pd

class Procrustes:
    @staticmethod
    def stretch_wide_children(df_: pd.DataFrame,
                              household_id_column: str = "id_household",
                              postfix_: str = "c") -> pd.DataFrame:
        """Transforms a specific dataframe from long to wide

        Parameters
        ----------
        df_
        household_id_column
        postfix_

        """
        df_ = df_.sort_values(["ordinal_person_age"], ascending=False)
        df_["index_"] = (df_.
                         groupby(household_id_column).
                         cumcount()
                         )
        df_ = df_.pivot(index=household_id_column, columns="index_")
        df_.columns = [f"{v1}_{postfix_}{v2}" for v1, v2 in df_.columns]
        return df_.reset_index()

    @staticmethod
    def stretch_wide_adults(df_: pd.DataFrame,
                            id_list: tuple[str, str] = ("id_household",
                                                             "id_person")
                                                             #"id_partner")
                            ) -> pd.DataFrame:
        # TODO add a check to make sure a person always has a partner
        #_household, _person, _partner = id_list
        _household, _person = id_list
        # hh comes first in the list

        #df_["aux_id1"] = df_.groupby([_household, _person, _partner]).cumcount()
        df_["aux_id1"] = df_.groupby([_household, _person]).cumcount()
        # auxiliary id#1
        # because we scale the population up there are multiple records
        # of the same household;
        # this new internal id allows to differentiate between these duplicates

        aux_df = df_[[_household, _person]].copy().drop_duplicates()
        aux_df["aux_id2"] = aux_df.groupby([_household]).cumcount()
        # auxiliary id#2
        # this is the sequential number of every individual
        # within a unique household

        df_ = (df_.
               merge(aux_df, on=[_household, _person]).
               #drop(columns=[_person, _partner]))
               drop(columns=[_person]))

        _valid_columns = [e for e in df_.columns if e not in ["id_household",
                                                              "aux_id1",
                                                              "aux_id2"]]

        df_ = df_.pivot(index=[_household, "aux_id1"],
                        columns=["aux_id2"],
                        values=_valid_columns)

        df_.columns = [f"{a}_a{b}" for a, b in df_.columns.to_flat_index()]

        return df_.reset_index().drop(columns="aux_id1")

    @staticmethod
    def stretch_long(df_: pd.DataFrame, _postfix = "a") -> pd.DataFrame:
        if _postfix not in ["a", "c"]:
            raise ValueError("Postfix must be 'a' or 'c'")
        # must contain a column `id_household` and a postfix matching the pattern `*_a[0-9]` or `*_c[0-9]`
        _column_list = df_.columns.to_list()
        _column_list.remove("id_household")
        # TODO all columns must be for individuals; add raise

        _column_list = [_c[:-1] for _c in _column_list]
        # assume single digit indices only
        # move from ["sex_a1", "age_a1"] to ["sex_a", "age_a"]
        return (pd.
                wide_to_long(df_,
                             set(_column_list),
                             i="id_household",
                             j="inter_household_id").
                reset_index().
                drop(columns=["inter_household_id"]).
                rename(columns={_c: _c[:-2] for _c in _column_list if _c.endswith("_a")})
                )

    @staticmethod
    def force_data_types(df_: pd.DataFrame,
                         map_: dict[str, dict[str, str]],
                         selection_: tuple[str, str]) -> pd.DataFrame:
        """Converts datatypes according to a set of rules"""
        for group_ in map_.keys():
            if group_ in selection_:
                for prefix_, dtype_ in map_[group_].items():
                    for c_ in df_.columns:
                        if c_.startswith(prefix_):
                            df_[c_] = df_[c_].astype(dtype_)

        return df_



