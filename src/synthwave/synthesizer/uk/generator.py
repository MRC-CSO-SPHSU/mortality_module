from sklearn.dummy import DummyClassifier
import logging

import pandas as pd
from tqdm.notebook import tqdm

from synthwave.utils.general import generate_household_id, \
    generate_personal_ids
from synthwave.utils.procrustes import Procrustes

logger = logging.getLogger(__name__)

import pyarrow as pa
from rdt.transformers import FloatFormatter
from sdv.single_table import CTGANSynthesizer
from synthwave.synthesizer.utils import metadata_constructor, clean_imputed_data
#from synthwave.utils.uk.alignment_census import map_qualification, get_coarse_mapping

from synthwave.synthesizer.uk.constraints import *
from sdv.constraints import create_custom_constraint_class
from synthwave.synthesizer.abstract.axolotl_tank import assign_household_id
import re
import yaml

import joblib

from synthwave.synthesizer.axolotl_tank import get_optimal_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from importlib.resources import files

logger.info("start")

HOUSEHOLD_ID_MAP = {_v: _i  for _i, _v in enumerate(["a0", "a1+", "c0", "c1", "c2", "c3+", "m2", "m3", "mc3", "m4", "mc4"])}
# their relative order is irrelevant at the moment as long as it is the same across all data

MAX_CHILDREN = yaml.safe_load(files("synthwave.data.understanding_society").joinpath('syntet.yaml').read_text())["MAX_CHILDREN"]

adults = pd.read_csv("~/Work/IHW/datasets/out10.csv", dtype_backend="pyarrow")

# FIXME this is a temporary workaround here to be able to employ constraints, it should be done elsewhere.
_adjusted_time = [_c for _c in adults.columns if "minutes" in _c]
adults[_adjusted_time] = adults[_adjusted_time].div(15).round(0).mul(15).astype("uint16[pyarrow]")

adults = clean_imputed_data(adults)

# TODO we leave job classification to simulated annealing and synthesis; the former does nation-wide adjustment, the latter restricts generation only to combinations that are already there.

# TODO imputation affects hourly rates, but we don't check them atm

adults["indicator_person_requires_correction"] = ~validate((
    "indicator_person_is_self_employed",
    "indicator_person_is_employed",

    "minutes_person_employment",
    "income_person_pay",
    "hours_person_overtime",

    "hours_person_self_employment",
    "income_person_self_employment",

    "category_person_job_nssec",
    "category_person_job_sic",

    "category_person_job_status",

    "income_person_second_job"), adults)

_trouble_counts = adults[adults["indicator_person_requires_correction"]][["id_person"]].value_counts().reset_index()

_people_counts = (adults[adults["id_person"].isin(adults[adults["indicator_person_requires_correction"]]["id_person"].unique())][["id_person"]].
                  value_counts().
                  reset_index().
                  rename(columns={"count": "total_individuals"}).
                  merge(_trouble_counts, on="id_person", how="inner").
                  rename(columns={"count": "corrupted_records"}).
                  assign(sampling_pool=lambda x: x.total_individuals - x.corrupted_records).
                  assign(pool_ratio=lambda x: x.sampling_pool / x.corrupted_records))

to_transplant = _people_counts[_people_counts["pool_ratio"].ge(0.1)]["id_person"]

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors

num_pipe = make_pipeline(MinMaxScaler())
cat_pipe = make_pipeline(OneHotEncoder(handle_unknown="error", sparse_output=False))

adults["id_person_new"] = range(1, len(adults) + 1)

preprocessor_tree = make_column_transformer(
    (num_pipe, ["minutes_person_employment", "income_person_pay", "hours_person_overtime",
                "hours_person_self_employment", "income_person_self_employment", "income_person_second_job"]),
    (cat_pipe, ["category_person_job_nssec", "category_person_job_sic", "category_person_job_status"]),
    remainder='passthrough',
    n_jobs=8,
    force_int_remainder_cols=False
)

_target_columns = ["indicator_person_is_self_employed", "indicator_person_is_employed",
                   "minutes_person_employment", "income_person_pay", "hours_person_overtime",
                   "hours_person_self_employment", "income_person_self_employment",
                   "category_person_job_nssec", "category_person_job_sic",
                   "category_person_job_status", "income_person_second_job"]
# run across *all* records
preprocessor_tree.fit(adults[_target_columns])

    # parallel
    # from multiprocessing import Pool
    # pool = Pool(8)
    #
    # # split into list of frames grouped by id
    # #
    #
    # # move all invalid records and donors based on the same id to another frame.
    # correction_df = adults[adults["id_person"].isin(to_transplant)].copy()
    #
    # # keep valid records in the old dataframe
    # ttt_adults = adults[~adults["id_person"].isin(to_transplant)].copy()
    #
    # clones = [g for _, g in correction_df.groupby('id_person')]
    # # NOTE can't believe there is no aesthetically more pleasant approach
    #
    # def surgery(_clones):
    #     _recipients = _clones[_clones["indicator_person_requires_correction"]][_target_columns + ["id_person_new"]]
    #     _donors = _clones[~_clones["indicator_person_requires_correction"]][_target_columns]
    #
    #     knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    #
    #     _ready_to_donor = preprocessor_tree.transform(_donors)
    #
    #     knn.fit(_ready_to_donor)
    #
    #     _final_donors = knn.kneighbors(preprocessor_tree.transform(_recipients.drop(columns=["id_person_new"])), return_distance=False)
    #
    #     _mask = _clones["id_person_new"].isin(_recipients["id_person_new"])
    #
    #     _transplants = _donors.iloc[_final_donors.flatten(), :].reset_index(drop=True)
    #     # TODO make a test to ensure this works as intended
    #
    #     _clones.loc[_mask, _target_columns] = _transplants[_target_columns].values
    #     # TODO see https://stackoverflow.com/questions/39267372/replace-rows-in-a-pandas-df-with-rows-from-another-df for inconsistent behaviour
    #     _clones.loc[_mask, "indicator_person_requires_correction"] = False
    #     return _clones
    #
    # a = [pool.apply_async(surgery, kwds={"_clones": clone_}) for clone_ in clones]
    #
    # for i in a:
    #     i.wait()

for _person in tqdm(to_transplant):
    _recipients = adults[adults["id_person"].eq(_person) &
                         adults["indicator_person_requires_correction"]][_target_columns + ["id_person_new"]]

    _donors = adults[adults["id_person"].eq(_person) &
                     ~adults["indicator_person_requires_correction"]][_target_columns]

    knn = NearestNeighbors(n_neighbors=1,
                           algorithm='ball_tree',
                           n_jobs=8)

    _ready_to_donor = preprocessor_tree.transform(_donors)

    knn.fit(_ready_to_donor)

    _final_donors = knn.kneighbors(preprocessor_tree.transform(_recipients.drop(columns=["id_person_new"])), return_distance=False)

    _mask = adults["id_person_new"].isin(_recipients["id_person_new"])

    _transplants = _donors.iloc[_final_donors.flatten(), :].reset_index(drop=True)
    # TODO make a test to ensure this works as intended

    adults.loc[_mask, _target_columns] = _transplants[_target_columns].values
    # TODO see https://stackoverflow.com/questions/39267372/replace-rows-in-a-pandas-df-with-rows-from-another-df for inconsistent behaviour
    adults.loc[_mask, "indicator_person_requires_correction"] = False
# NOTE this is a pretty crude version of what should be; it leads to excessive repetitions in the block of given variables. more variations/iterations of imputations should alleviate this

for c_ in adults.columns:
    # this approach keeps zeroes and doesn't affect small incomes at this stage any more
    if "income" in c_:
        positive_income = adults[c_].le(10) & adults[c_].gt(0)
        adults.loc[positive_income, c_] = 10

        negative_income = adults[c_].ge(-10) & adults[c_].lt(0)
        adults.loc[negative_income, c_] = -10

        adults[c_] = adults[c_].round(-1)
        # TODO investigate how early coarse-graining of income affects the transplant stage

# FIXME removing records that haven't been processed properly is a temporary workaround
_unfixable_individuals = adults[adults["indicator_person_requires_correction"]]["id_person"]
_unfixable_households = adults[adults["id_person"].isin(_unfixable_individuals)]["id_household"]
adults = adults[~adults["id_household"].isin(_unfixable_households)]

adults = adults.drop(columns=["indicator_person_requires_correction", "id_person_new"])

# NOTE even though the imputation stage might introduce some (improbable) combinations of parameters that do not exist in reality we disregard them. The process of annealing will alleviate this.

adults.loc[adults["category_household_type"].isin(['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a9',]), "category_household_type"] = "a1+"
adults.loc[adults["category_household_type"].isin(['c3', 'c4', 'c5', 'c6', 'c7', 'c8']), "category_household_type"] = "c3+"

# m2 is 1 adult and 1 pseudo-adult, no couples
# m3 and m4 can have one couple inside, their records must go first, make sure the ages are ordered and pass the check

def mark_households_with_couples(_df: pd.DataFrame, _hh_type: str) -> pd.Series:

    if _hh_type not in (_types := ["m3", "m4"]):
        raise ValueError(f"Household type must be either {_types[0]} or {_types[1]}")

    _no_couples_mask = _df[_df["category_household_type"].eq(_hh_type)][["id_household", "id_person", "id_partner"]].drop_duplicates().groupby(["id_household"])["id_partner"].apply(lambda x: x.isna().all()).reset_index()
    # this way we mark m3 households with no couples inside i.e., one adult and two pseudo-adults
    _couples = _no_couples_mask[~_no_couples_mask["id_partner"]]["id_household"]
    return _df["id_household"].isin(_couples)


adults.loc[mark_households_with_couples(adults, "m3"), "category_household_type"] = "mc3"
adults.loc[mark_households_with_couples(adults, "m4"), "category_household_type"] = "mc4"

# mf is mix and match
# total_individuals == 3, total_children == 1 # that is 1 adult and 1 pseudo-adult, otherwise that would be a couple
# total_individuals == 4, total_children == 1 # can be a couple inside + 1 pseudo-adult; or 1 adult and two pseudo-adults; needs a split
# TODO do the same for mf

#_test = adults[adults["category_household_type"].isin(["mf"]) & adults["total_adults"].eq(2) & adults["total_children"].eq(1)].copy()

class Syntets:
    DEFAULT_GROUPS = ["a0", "a1+", "c0", "c1", "c2", "c3+", "m2", "m3", "mc3", "m4", "mc4"]
    def __init__(self, raw_data):
        self.raw_data = raw_data.copy()
        self.data = self.raw_data.copy()
        self.locations = raw_data["category_household_location"].unique()
        self.household_types = raw_data["category_household_type"].unique()

        self.groups = dict()
        self.subsets = []

    @staticmethod
    def generate_schema(_df: pd.DataFrame):
        _s = []

        for _c in _df.columns:
            if _c.startswith(("indicator_", "mlb_")):
                _s.append((_c, pa.bool_()))
            if _c.startswith(("ordinal_", "category_", "hours_")):
                _s.append((_c, pa.uint8()))
            if _c.startswith("minutes_"):
                _s.append((_c, pa.uint16()))
            if _c.startswith("income_"):
                if _c == "income_person_self_employment":
                    _s.append((_c, pa.int16()))
                elif _c == "income_person_investment":
                    _s.append((_c, pa.uint32()))
                else:
                    _s.append((_c, pa.uint16()))
        return pa.schema(_s)

    def convert_types(self):
        # unfortunately, ctgan still struggles to deal well with modern types, we replace all of them
        for _g, _r in self.subsets:
            _columns = self.groups[(_g, _r)]["data"].columns
            _int_target = []

            for _c in _columns:
                if _c.split("_")[0] in ["category", "hours", "ordinal", "income", "ordinal", "total", "minutes"] and _c != "category_household_type":
                    _int_target.append(_c)

            self.groups[(_g, _r)]["data"][_int_target] = self.groups[(_g, _r)]["data"][_int_target].astype(int)

            _bools = [_c for _c in _columns if _c.startswith(("indicator_", "mlb_"))]
            self.groups[(_g, _r)]["data"][_bools] = self.groups[(_g, _r)]["data"][_bools].astype(bool)

    def split_data(self):
        for _g in Syntets.DEFAULT_GROUPS:
            for _r in self.locations:
                d = self._splitter(self.data, [_g], _r)
                if not _g.endswith("+"):
                    d = d.drop(columns=["total_children"])
                if len(d) > 0:
                        self.subsets.append((_g, _r))
                        self.groups[(_g, _r)] = {"data": d, "model": None, "dropouts": None}

    @staticmethod
    def _splitter(_df: pd.DataFrame, _type: list, _location: int) -> pd.DataFrame:
        # select a subset
        return (_df[_df["category_household_type"].isin(_type) &
                    _df["category_household_location"].eq(_location)]
                .copy()
                .drop(columns=["category_household_type",
                               "category_household_location",
                               ]).reset_index(drop=True))
    # we don't need household type any more, another hh classification will be introduced at the post-synthesis stage

    def locate_degenerate_distributions(self):
         # degeneracy might appear when we convert from long to wide
        for _g, _r in self.subsets:
            if self.groups[(_g, _r)]["data"] is None:
                logger.warning(f"Dataset {_g} for region {_r} is missing")
                continue
            else:
                _combinations = self.groups[(_g, _r)]["data"].apply(pd.unique)
                _exclude = _combinations[_combinations.map(len) == 1].map(lambda x: x[0]).to_dict()
                self.groups[(_g, _r)]["dropouts"] = _exclude
                self.groups[(_g, _r)]["data"].drop(columns=_exclude.keys(), inplace=True)

                _cmb = _combinations[_combinations.index.str.startswith("category_") &
                                     _combinations.map(len).eq(2)]
                if len(_cmb) > 0:
                    _msg = f"In region {_r} and household {_g} columns {_cmb.index} are potential candidates for remapping"
                    logger.warning(_msg)

    @staticmethod
    def _get_household_attributes(_df: pd.DataFrame) -> list[str, ...]:
        _columns = []
        for _c in _df.columns:
            if "household" in _c and not _c.startswith("id_"):
                _columns.append(_c)
        return _columns

    def restructure_data(self):
        for _g, _r in self.subsets:

            _columns_household = self._get_household_attributes(self.groups[(_g, _r)]["data"])

            if _g.endswith("+"):
                _columns_household.append("total_children")

            household_attributes = self.groups[(_g, _r)]["data"][_columns_household + ["id_household"]].copy().drop_duplicates()

            self.groups[(_g, _r)]["data"].drop(columns=_columns_household, inplace=True)

            if not _g.startswith("a"):
                if _g.startswith("c"):
                    self.groups[(_g, _r)]["data"].sort_values(by=["id_household",
                                                                  'ordinal_person_age', # now first person is always younger, but sex can vary
                                                                  'indicator_person_sex'], inplace=True)
                elif _g.startswith("m"):
                    if "c" in _g:
                        self.groups[(_g, _r)]["data"]["is_in_couple"] = self.groups[(_g, _r)]["data"]["id_partner"].notna()
                        self.groups[(_g, _r)]["data"] = self.groups[(_g, _r)]["data"].drop(columns="id_partner")
                        self.groups[(_g, _r)]["data"] = Procrustes.stretch_wide_adults(self.groups[(_g, _r)]["data"].
                                                                                       sort_values(["id_household", "is_in_couple", "ordinal_person_age"], ascending=False).
                                                                                       drop(columns="is_in_couple"))
                        # for mc3/mc4 the rules are as follows:
                        #  - the eldest person in couple comes first
                        #  - the other person comes second (obviously)
                        #  - the age of a0 >= the age of a1, we check for it
                        #  - we don't add any more age constraints; the third person can be another adult of some age; their number is small but not negligible.
                    else:
                        # for m3/m4 we sort by age and sex, this is the only reasonable assumption here
                        # - age of a0 >= age of a1 >= age of a2
                        self.groups[(_g, _r)]["data"] = Procrustes.stretch_wide_adults(self.groups[(_g, _r)]["data"].
                                                                                       sort_values(["id_household", "ordinal_person_age", "indicator_person_sex"], ascending=False))
                else:
                    pass # TODO make sure the code is adjusted for mf

            self.groups[(_g, _r)]["data"] = self.groups[(_g, _r)]["data"].merge(household_attributes, how="left", on="id_household")
            # we store the household attributes in a separate dataframe and merge it with the transformed one
            # we could do this for transformed tables only, but the performance hit is negligible

    def drop_id_columns(self):
        # we keep this bit in a separate function because we need id_household for children
        for _g, _r in self.subsets:
            _columns_id = [_c for _c in self.groups[(_g, _r)]["data"] if _c.startswith("id_")]
            self.groups[(_g, _r)]["data"].drop(columns=_columns_id, inplace=True)

    @staticmethod
    def _get_postfixes(_df) -> list[str] | list[str, ...]:
        _p = [re.search(r"\_a[0-9]$", _c) for _c in _df.columns]
        _p = list(set(_p))
        _p.remove(None)

        if len(_p) > 0:
            return list(set([_match[0] for _match in _p]))
        else:
            return [""]

    @staticmethod # TODO consolidate code
    def _get_postfixes_children(_df) -> list[str] | list[str, ...]:
        _p = [re.search(r"\_c[0-9]$", _c) for _c in _df.columns]
        _p = list(set(_p))
        _p.remove(None)

        if len(_p) > 0:
            return list(set([_match[0] for _match in _p]))
        else:
            return [""]

    def init_models(self, _use_cuda=False, _batch_size=5000, _epochs=5000):
        for _g, _r in self.subsets:
            self.groups[(_g, _r)]["model"] = CTGANSynthesizer(metadata_constructor(self.groups[(_g, _r)]["data"], f"{_g}_{_r}"),
                                                              enforce_rounding=False,
                                                              epochs=_epochs,
                                                              verbose=True,
                                                              cuda=_use_cuda,
                                                              batch_size=_batch_size
                                                              )
            self.groups[(_g, _r)]["model"].validate(self.groups[(_g, _r)]["data"])
            self.groups[(_g, _r)]["model"].auto_assign_transformers(self.groups[(_g, _r)]["data"])

            for _c, _t in self.groups[(_g, _r)]["model"].get_transformers().items():
                if _t is not None:
                    _repr = _t.computer_representation
                    self.groups[(_g, _r)]["model"].update_transformers(column_name_to_transformer={_c: FloatFormatter(enforce_min_max_values=True,
                                                                                                                      computer_representation=_repr)})

    def attach_constraints(self):

        def get_increments(_column_name, _increment):
            # increments: income and minutes
            return {
                'constraint_class': 'FixedIncrements',
                'constraint_parameters': {
                    'column_name': _column_name,
                    'increment_value': _increment
                }
            }

        for _g, _r in self.subsets:
            # these are general constraints for every column of the kind
            # because we use minutes instead of hours we can't say income is always >= time due to minimal rate;
            # we can't also say the opposite due to variable rates
            for _c in self.groups[(_g, _r)]["data"].columns:
                if "minutes" in _c:
                    self.groups[(_g, _r)]["model"].add_constraints([get_increments(_c, 15)])
                if "income" in _c:
                    self.groups[(_g, _r)]["model"].add_constraints([get_increments(_c, 10)])
        # TODO we can have a custom constraint for income rate?

        def get_combinations(_columns: list) -> dict:
            return {
                'constraint_class': 'FixedCombinations',
                'constraint_parameters': {
                    'column_names': _columns
                }
            }

        def get_inequality(_columns: list, _table_name: str) -> dict:
            return {
                'constraint_class': 'Inequality',
                'table_name': _table_name,
                'constraint_parameters': {
                    'low_column_name': _columns[0],
                    'high_column_name': _columns[1],
                    'strict_boundaries': False
                }
            }

        for _g, _r in self.subsets:
            postfixes = self._get_postfixes(self.groups[(_g, _r)]["data"])

            # Age constraints where possible
            if _g.startswith(("c", "m")): # this constraint always works because we sort the data in advance
                self.groups[(_g, _r)]["model"].add_constraints(constraints=[
                    get_inequality(["ordinal_person_age_a0",
                                    "ordinal_person_age_a1"], f"{_g}_{_r}"),
                ])
            if _g in ["m3", "m4"]:
                self.groups[(_g, _r)]["model"].add_constraints(constraints=[
                    get_inequality(["ordinal_person_age_a1",
                                    "ordinal_person_age_a2"], f"{_g}_{_r}"),
                ])
            if _g == "m4":
                self.groups[(_g, _r)]["model"].add_constraints(constraints=[
                    get_inequality(["ordinal_person_age_a2",
                                    "ordinal_person_age_a3"], f"{_g}_{_r}"),
                ])

            if _g.startswith("c") or _g in ["mc3", "mc4"]:
                self.groups[(_g, _r)]["model"].add_constraints(constraints=[
                    # category_person_legal_marital_status in couples should have a fixed number of combinations;
                    # this works for any household with a couple inside by data design
                    get_combinations(["category_person_legal_marital_status_a0",
                                      "category_person_legal_marital_status_a1"])
                ])

            # add to the list of constraint classes
            for klass in [MetaEmployment, BenefitsIncome, MetaEmploymentNoSecondJob]:
                self.groups[(_g, _r)]["model"].add_custom_constraint_class(create_custom_constraint_class(klass.is_valid,
                                                                                                          klass.transform,
                                                                                                          klass.reverse_transform),
                                                                           klass.__name__)

            for _p in postfixes:
                # TODO add a check to make sure all columns are in there. should never happen
                # TODO order of constraints matters; do some tests
                # we repeat this for every adult in the household
                self.groups[(_g, _r)]["model"].add_constraints([
                    get_combinations([_c for _c in self.groups[(_g, _r)]["data"].columns if "qualification" in _c and _c.endswith(_p)]),
                    # the number of columns can vary from group to group
                    # FIXME the benefits have been corrupted by imputation and therefore do not pass the constraint check
                ])
                if "income_person_second_job" + _p in  self.groups[(_g, _r)]["data"].columns:
                    self.groups[(_g, _r)]["model"].add_constraints([
                                            MetaEmployment.get_schema(
                        ["indicator_person_is_self_employed" + _p,
                         "indicator_person_is_employed" + _p,

                         "minutes_person_employment" + _p,
                         "income_person_pay" + _p,
                         "hours_person_overtime" + _p,

                         "hours_person_self_employment" + _p,
                         "income_person_self_employment" + _p,

                         "category_person_job_nssec" + _p,
                         "category_person_job_sic" + _p,

                         "category_person_job_status" + _p,

                         "income_person_second_job" + _p])])
                else:
                    self.groups[(_g, _r)]["model"].add_constraints([
                                            MetaEmploymentNoSecondJob.get_schema(
                        ["indicator_person_is_self_employed" + _p,
                         "indicator_person_is_employed" + _p,

                         "minutes_person_employment" + _p,
                         "income_person_pay" + _p,
                         "hours_person_overtime" + _p,

                         "hours_person_self_employment" + _p,
                         "income_person_self_employment" + _p,

                         "category_person_job_nssec" + _p,
                         "category_person_job_sic" + _p,

                         "category_person_job_status" + _p])])

    def train(self, save_path, verbose=False):
        for _g, _r in self.subsets:
            if verbose:
                print(_g, _r)
                print(self.groups[(_g, _r)]["dropouts"])
                print(len(self.groups[(_g, _r)]["data"]))

            self.groups[(_g, _r)]["model"].fit(self.groups[(_g, _r)]["data"])
            self.groups[(_g, _r)]["model"].save(filepath=save_path + f'model_{_g}_{_r}.pkl')

            with open(save_path + f'dropouts_{_g}_{_r}.yaml', 'w') as yml:
                yaml.dump(self.groups[(_g, _r)]["dropouts"], yml, allow_unicode=True)

    @staticmethod
    def _train_children(_data_household: pd.DataFrame,
                        _max_household_children: int,
                        _household_type: str,
                        _household_location: int,
                        _model_location: str):
        """ Trains RF to predict basic attributes of children for a given household type.

        We assume that data has been cleaned and merged together as follows:
        | a0 | ... | c0 | ...

        There is no ids whatsoever since we don't need them for prediction.

        The provided dataset must have at least one child per household with postfixes "_c0" and so on depending on the total number of children.
        Zero-order approximation is to have only attributes of adults for prediction. We select the best of them to predict the age of the oldest child,
        the next step is to add "ordinal_person_age_c0" to the zero-order approximation list of predictors and repeat the procedure of selection but now for the sex of the oldest child.
        We keep going in this pseudo-auto-regressive fashion until we can predict all attributes of the oldest child. If there is, for example, one more kid in the household we use info about adults *and* their older sibling for prediction now.

        Parameters
        ----------
        _data_household
        _max_household_children

        Returns
        -------

        """
        # TODO investigate too young (too old?) parents
        _full_base_predictors = [_c for _c in _data_household.columns if not _c.endswith(tuple(f"_c{_id}" for _id in range(_max_household_children)))]
        _extra_children_predictors = []

        _model_collection = {"children":
                                 {"total": {_max_household_children:
                                                {"id": {_child_id: {"age": tuple(),
                                                                    "sex": tuple(),
                                                                    "ethnic_group": tuple()} for _child_id in range(_max_household_children)}}}}}

        for aux_id in range(_max_household_children):
            _target_map = {
                "age": f"ordinal_person_age_c{aux_id}",
                "sex": f"indicator_person_sex_c{aux_id}",
                "ethnic_group": f"category_person_ethnic_group_c{aux_id}"
            }
            # TODO wait for Python 3.14 to employ string templates

            def _generate_model(_target_code):
                _set = _data_household[_target_map[_target_code]].unique()

                if len(_set) == 1:
                    # degeneracy
                    _dummy = DummyClassifier(strategy="constant", constant=_set[0])
                    _dummy.fit(_data_household[_full_base_predictors], _data_household[_target_map[_target_code]])
                    return _dummy, _full_base_predictors

                # TODO rf is better than knn in that it can work fine without one hot encoding. still, current implementation does see *all* columns as numeric. works well though. see https://github.com/scikit-learn/scikit-learn/pull/12866
                _rf = RandomForestClassifier(n_estimators=1024,
                                 criterion='entropy',
                                 max_depth=None,
                                 max_features=None,
                                 random_state=1,
                                 n_jobs=4,
                                 )

                X_train, X_test, y_train, y_test = train_test_split(_data_household[_full_base_predictors + _extra_children_predictors],
                                                                    _data_household[_target_map[_target_code]],
                                                                    stratify=_data_household[_target_map[_target_code]],
                                                                    random_state=42, test_size=0.1)
                _optimal_predictors = get_optimal_features(_data_household[_full_base_predictors + _extra_children_predictors],
                                                           X_train, X_test, y_train, y_test)

                _rf.fit(_data_household[_optimal_predictors],
                        _data_household[_target_map[_target_code]])

                joblib.dump(_rf, _model_location + f'{_household_type}_{_household_type}_tc{_max_household_children}_cid{aux_id}_{_target_code}.joblib')

                with open(_model_location + f'{_household_type}_l{_household_location}_tc{_max_household_children}_cid{aux_id}_{_target_code}.yaml', 'w') as yml:
                    yaml.dump(_optimal_predictors, yml, allow_unicode=True)

                _extra_children_predictors.append(_target_map[_target_code])
                # only include if the target is not degenerate

                return _rf, _optimal_predictors

            for _k, _v in _target_map.items():
                _model_collection["children"]["total"][_max_household_children]["id"][aux_id][_k] = _generate_model(_k)

        return _model_collection

    def train_children(self, _df_children: pd.DataFrame, save_path: str = "/tmp/", verbose = False):
        # TODO code consolidation
        for _g, _r in self.subsets:
            if _g in ["a1+", "c1", "c2", "c3+"]: # TODO mf
                if verbose:
                    print(f"{_g=}, {_r=}")
                if _g.endswith("+"):
                    _counts = self.groups[(_g, _r)]["data"]["total_children"].unique()
                    for _total_children in _counts:
                        _df = self.groups[(_g, _r)]["data"].copy()
                        _df = _df[_df["total_children"].eq(_total_children)].drop(columns="total_children")

                        _columns_id = [_c for _c in _df if _c.startswith("id_")]
                        _columns_id.remove("id_household")
                        _df.drop(columns=_columns_id, inplace=True)

                        if _total_children > MAX_CHILDREN:
                            continue # if too big - skip
                        else:
                            _hh = _df["id_household"].unique()
                            _children = _df_children[_df_children["id_household"].isin(_hh)].copy()

                        if _total_children == 1:
                            _children = _children.rename(columns={_c: _c + "_c0" for _c in _children.columns if not _c.startswith("id_")})
                        else:
                            _children = Procrustes.stretch_wide_children(_children)

                        _df = _df.merge(_children,
                                        how="inner", # inner merge to avoid incomplete records
                                        on="id_household").drop(columns=["id_household"], errors="ignore")

                        comb = _df.apply(pd.unique)
                        exclude = comb[comb.map(len) == 1].map(lambda x: x[0]).to_dict().keys()
                        exclude = [_e for _e in exclude if not _e.endswith(tuple(self._get_postfixes_children(_children)))]
                        # do not drop out degenerate attributes of children
                        _df.drop(columns=exclude, inplace=True)
                        # we do this cleaning because previous data manipulations i.e. merging and such might result in fewer records and therefore lower data variation.

                        #print(_df.apply(pd.unique))
                        _models = self._train_children(_data_household = _df,
                                                       _max_household_children = _total_children,
                                                       _household_type = _g,
                                                       _household_location = _r,
                                                       _model_location = save_path)

                        if "children" in self.groups[(_g, _r)].keys():
                            self.groups[(_g, _r)]["children"]["total"][_total_children] = _models["children"]["total"][_total_children]
                        else:
                            self.groups[(_g, _r)]["children"] = _models["children"]

                else:
                    # all households have the same number of children
                    assert "total_children" not in self.groups[(_g, _r)]["data"].columns

                    _df = self.groups[(_g, _r)]["data"].copy()

                    _columns_id = [_c for _c in _df if _c.startswith("id_")]
                    _columns_id.remove("id_household")
                    _df.drop(columns=_columns_id, inplace=True)

                    #infer the number of children
                    _total_children = int(''.join(filter(str.isdigit, _g)))

                    if _total_children > MAX_CHILDREN:
                            continue # if too big - skip
                    else:
                        _hh = _df["id_household"].unique()
                        _children = _df_children[_df_children["id_household"].isin(_hh)].copy()

                        if _total_children == 1:
                            _children = _children.rename(columns={_c: _c + "_c0" for _c in _children.columns if not _c.startswith("id_")})
                        else:
                            _children = Procrustes.stretch_wide_children(_children)

                        _df = _df.merge(_children,
                                        how="inner", # inner merge to avoid incomplete records
                                        on="id_household").drop(columns=["id_household"], errors="ignore")

                        comb = _df.apply(pd.unique)
                        exclude = comb[comb.map(len) == 1].map(lambda x: x[0]).to_dict().keys()
                        exclude = [_e for _e in exclude if not _e.endswith(tuple(self._get_postfixes_children(_children)))]
                        # do not drop out degenerate attributes of children
                        _df.drop(columns=exclude, inplace=True)

                        _models = self._train_children(_data_household = _df,
                                                       _max_household_children = _total_children,
                                                       _household_type = _g,
                                                       _household_location = _r,
                                                       _model_location = save_path)

                        self.groups[(_g, _r)]["children"] = _models["children"]

    @staticmethod
    def pad_children(_df: pd.DataFrame) -> pd.DataFrame:
        _personal_indicators = {_c: False for _c in _df.columns if _c.startswith(("mlb", "indicator_person_vocational_qualification", "indicator_person_is"))}

        _income = {_c: 0 for _c in _df.columns if _c.startswith("income")}

        _time = {_c: 0 for _c in _df.columns if _c.startswith("hours", "minutes")}

        _ghq = {_c: 0 for _c in _df.columns if "ghq" in _c}

        _job_status = {_c: 3 for _c in _df.columns if "job_status" in _c}

        _job_class = {_c: 0 for _c in _df.columns if any(x in _c for x in ["nssec", "job_sic"])}

        _marital = {_c: 1 for _c in _df.columns if "marital" in _c}

        _full_qualification = {_c: 96 for _c in _df.columns if "full_highest" in _c}

        _sf12 = {_c: 0 for _c in _df.columns if "_sf" in _c}

        _misc = {_c: 0 for _c in ["ordinal_person_financial_situation", "ordinal_person_life_satisfaction",
                                  "category_person_father_education", "category_person_mother_education"]}
        # NOTE education of parents is tricky as it involves some temporal components
        _df = _df.fillna(value=_personal_indicators | _income | _time | _ghq | _job_status | _job_class | _marital | _full_qualification | _sf12 | _misc)

        _household_attributes = ["category_household_location", "category_household_type",
                                 "ordinal_household_total_cars", "category_household_house_ownership",
                                 "indicator_household_has_central_heating", "total_children"]

        _df[_household_attributes] = _df[_household_attributes].groupby(["id_household"],
                                                                        sort=False)[_household_attributes].apply(lambda x: x.ffill().bfill())
        return _df

    def add_children(self,
            _df: pd.DataFrame,
                         _max_household_children: int,
                         _household_type: str,
                         _household_location: int,
                         _model_location: str,
                         _mini_batch_id_: int = None,
                         _micro_batch_id_: int = None,
                         ) -> pd.DataFrame:

        # this is the highest level function
        """Adds children to provided households

        Parameters
        ----------
        _df
        _max_household_children
        _household_type
        _household_location
        _model_location
        _mini_batch_id_
        _micro_batch_id_

        Returns
        -------

        """
        assert _household_type in ["a1+", "c1", "c2", "c3+"]

        # MAKE CHILDREN
        _children_attributes = dict()
        _local_df = _df.copy()

        for aux_id in range(_max_household_children):

            _target_map = {
                    "age": f"ordinal_person_age_c{aux_id}",
                    "sex": f"indicator_person_sex_c{aux_id}",
                    "ethnic_group": f"category_person_ethnic_group_c{aux_id}"
            }

            # loop over target map k, v
            for _k, _v in _target_map.items():
                # load yaml with optimal predictors
                with open(_model_location + f'{_household_type}_l{_household_location}_tc{_max_household_children}_cid{aux_id}_{_k}.yaml', 'w') as stream:
                    _optimal_predictors = yaml.safe_load(stream)

                # check if any of the siblings is there if yes attach corresponding series
                for _c in _optimal_predictors:
                    if _c in _children_attributes.keys():
                        _local_df = pd.concat([_local_df, _children_attributes[_c]], ignore_index=True,  axis=1)

                # load the model itself
                _model = joblib.load(_model_location + f'{_household_type}_{_household_type}_tc{_max_household_children}_cid{aux_id}_{_k}.joblib')
                _children_attributes[_v] = pd.Series(_model.predict(_local_df[_optimal_predictors]), name=_v)
                # predict attributes, store in a series, append to the list

        _children = pd.concat(_children_attributes.values().to_list(), ignore_index=True,  axis=1)
        _children["id_household"] = _df["id_household"]

        if _max_household_children == 1:
            _children = _children.rename(columns={_c: _c[:-3] for _c in _children.columns if not _c.startswith("id_")})
        else:
            _children = Procrustes.stretch_long(_children, "c")
            # TODO code consolidation

        if _household_type != "a1+":
            # for one adult no need to transform the table
            _df = Procrustes.stretch_long(_df, "a")

        _df = pd.concat([_df, _children])

        _df = generate_personal_ids(_df, contains_couples=False if _household_type == "a1+" else True)

        return self.pad_children(_df)

    def generator(self,
                  _local_model: CTGANSynthesizer,
                  _household_type: str,
                  _household_location: int,
                  _local_dropouts: dict,
                  _mini_batch_id: int,
                  _micro_batch_id: int,
                  _total_rows: int = 10_000_000,
                  _batch_size: int = 1_000_000):

        synthetic_data = _local_model.sample(num_rows=_total_rows, batch_size=_batch_size)
        # assign household ids
        synthetic_data["id_household"] = generate_household_id(_sample_size = len(synthetic_data),
                                                               _mini_batch_id = _mini_batch_id,
                                                               _micro_batch_id = _micro_batch_id,
                                                               _region_id = _household_location,
                                                               _household_type = HOUSEHOLD_ID_MAP[_household_type])

        # reinstate dropouts
        synthetic_data["category_household_location"] = _household_location
        for column_name, column_value in _local_dropouts:
            synthetic_data[column_name] = column_value
            # we re-introduce degenerate columns here to make sure all tables have the same structure

        # TODO add columns for Census matching
        # this block covers *all* possible combinations
        if _household_type.startswith("a"):
            if _household_type.endswith("0"):
                # one adult, no children
                synthetic_data = generate_personal_ids(synthetic_data, contains_couples=False)
            else:
                # one adult, several children
                _split = []
                for _children in synthetic_data["total_children"].unique():
                    # loop over children in a1+
                    _split.append(
                        self.add_children(_df=synthetic_data[synthetic_data["total_children"] == _children],
                                      _max_household_children = _children,
                                      _household_type = _household_type,
                                      _household_location = _household_location,
                                      _model_location = "/tmp", # FIXME no hardcoded values
                                      _mini_batch_id_ = _mini_batch_id,
                                      _micro_batch_id_ = _micro_batch_id)
                    )
                synthetic_data = pd.concat(_split)

        elif _household_type.startswith("c"):
            if _household_type.endswith("0"):
                # couple, no children
                synthetic_data = generate_personal_ids(synthetic_data, contains_couples=True)
                synthetic_data = Procrustes.stretch_long(synthetic_data, "a")
            else:
                # couple, several children
                if _household_type in ["c1", "c2"]:
                    synthetic_data = self.add_children(_df=synthetic_data,
                                      _max_household_children = int(''.join(filter(str.isdigit, _household_type))),
                                      _household_type = _household_type,
                                      _household_location = _household_location,
                                      _model_location = "/tmp", # FIXME no hardcoded values
                                      _mini_batch_id_ = _mini_batch_id,
                                      _micro_batch_id_ = _micro_batch_id)
                else:
                    # loop over children c3+
                    _split = []
                    for _children in synthetic_data["total_children"].unique():
                        # loop over children in a1+
                        _split.append(
                            self.add_children(_df=synthetic_data[synthetic_data["total_children"] == _children],
                                          _max_household_children = _children,
                                          _household_type = _household_type,
                                          _household_location = _household_location,
                                          _model_location = "/tmp", # FIXME no hardcoded values
                                          _mini_batch_id_ = _mini_batch_id,
                                          _micro_batch_id_ = _micro_batch_id)
                        )
                    synthetic_data = pd.concat(_split)

        elif _household_type.startswith("mc"):
            # couple + some other people
            synthetic_data = generate_personal_ids(synthetic_data, contains_couples=True)
        else: # TODO this only works for m3, m4 not mf
            synthetic_data = generate_personal_ids(synthetic_data, contains_couples=False)

        return synthetic_data

    @staticmethod
    def assign_matching_columns(_data):
        pass
        #TODO finish this one
