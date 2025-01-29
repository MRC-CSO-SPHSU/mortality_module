import pandas as pd

from mortality_module.synthesizer.abstract.constraints import CustomConstraint

TARGET_ORDER = (
    ("indicator_person_is_self_employed", 0),
    ("indicator_person_is_employed", 1),

    ("minutes_person_employment", 2),
    ("income_person_pay", 3),
    ("hours_person_overtime", 4),

    ("hours_person_self_employment", 5),
    ("income_person_self_employment", 6),

    ("category_person_job_nssec", 7),
    ("category_person_job_sic", 8),

    ("category_person_job_status", 9),

    ("income_person_second_job", 10)
)

def filler(_df, _column_list, _indicator_column: pd.Series):
    # we pass a Series to be able to do unemployed; they are neither e not se
    typical_values = [round(_df[~_indicator_column][_c].median()) + 1 for _c in _column_list]
    _df.loc[_indicator_column, _column_list] = typical_values
    return _df

def _parameter_check(_columns, _order=TARGET_ORDER):
    if len(_columns) != len(_order):
        raise ValueError(f"Must be {len(_order)} but got"
                         f" {len(_columns)} instead")

    for _name, _index in _order:
        if _name not in _columns[_index]: # this less strict comparison allows for postfixes
            raise ValueError(f"Column {_index} must be {_name} but got"
                             f" {_columns[_index]}")


def validate(column_names: tuple[str, ...], _df: pd.DataFrame,
             has_second_job: bool = True) -> pd.Series:
    """ Validates all records in the table to make sure they follow a pattern.

    column_names is the list of column names to validate. Due to some external
    constraints the order of names must be fixed and exactly the same every
     time.

    The logic of this validation is as follows:
        - when a person is self-employed the corresponding flag is true, and the
         employment flag is false. That person must work at least some time
         meaning self-employment hours are always positive. At the same time the
         nature of this activity not always results with income, and sometimes
         can even incur some losses, so there is no income restriction.
         Also, being self-employed means a person can't be employed at all, so
         all corresponding employment attributes are zero. Finally, the job
         classes can't be zero, and the job status must not be 2 (employed). We
         don't care about their second job.

        - for an employed person the employment flag is true, and the
         self-employment one is false. This person must work a bit and earn
         some, but we disregard any potential overtime. All self-employment
         attributes are zero, job codes are not zero, and the job status is not
         1. Again, having another job is optional.

        - unemployed people must have both employment and self-employment flags
         set to false, all income and time attributes are zero, including second
         job income. Both job classes are zero, and the job status is not 1, 2.

    The output series is an XOR combination of all three checks. The data is
    designed and pre-processed in such a way that a simple OR should be enough,
    but we prefer to stay safe.

    Parameters
    ----------
    has_second_job
    column_names
    _df

    Returns
    -------

    """
    if has_second_job:
        _parameter_check(column_names)
    else:
        _parameter_check(column_names, TARGET_ORDER[:-1])

    _cn = list(column_names)

    _true_self_employed = (_df[_cn[:2]].isin([True, False]).all(axis=1) &
                           _df[_cn[2:5]].eq(0).all(axis=1) &
                           _df[_cn[5]].gt(0) &
                           _df[_cn[7:9]].ne(0).all(axis=1) &
                           _df[_cn[9]].ne(2)
                           )

    _true_employed = (_df[_cn[:2]].isin([False, True]).all(axis=1) &
                      _df[_cn[2:4]].gt(0).all(axis=1) &
                      _df[_cn[5:7]].eq(0).all(axis=1) &
                      _df[_cn[7:9]].ne(0).all(axis=1) &
                      _df[_cn[9]].ne(1)
                      )

    _unemployed = (~_df[_cn[:2]].any(axis=1) &
                   _df[_cn[2:5]].eq(0).all(axis=1) &
                   _df[_cn[5:7]].eq(0).all(axis=1) &
                   _df[_cn[7:9]].eq(0).all(axis=1) &
                   ~_df[_cn[9]].isin([1, 2])
                   )

    if has_second_job:
        _unemployed &= _df[_cn[10]].eq(0)

    return _true_self_employed ^ _true_employed ^ _unemployed


class MetaEmployment(CustomConstraint):
    @staticmethod
    def is_valid(column_names, data):
        return validate(column_names, data)

    @staticmethod
    def transform(column_names, data):
        """Replaces structural zeroes in income and time with corresponding medians"""
        # TODO investigate full vs partial median
        # TODO investigate if copy is needed
        _parameter_check(column_names)

        _cn = list(column_names)

        transformed_data = data.copy()

        # self-employed
        transformed_data = filler(transformed_data, _cn[2:5],
                                  transformed_data[column_names[0]])

        # employed
        transformed_data = filler(transformed_data, _cn[5:7],
                                  transformed_data[column_names[1]])

        # unemployed
        transformed_data = filler(transformed_data, _cn[2:7] + [_cn[10]],
                                  ~(transformed_data[column_names[0]] |
                                    transformed_data[column_names[1]]))

        return transformed_data

    @staticmethod
    def reverse_transform(column_names, data):
        """Does the inverse transformation,
         pads corresponding time and income columns for non-(self)employed
         with zeroes"""

        _parameter_check(column_names)

        _cn = list(column_names)

        reversed_data = data.copy()

        # self-employed
        reversed_data.loc[reversed_data[column_names[0]], _cn[2:5]] = 0

        # employed
        reversed_data.loc[reversed_data[column_names[1]], _cn[5:7]] = 0

        # unemployed
        reversed_data.loc[~(reversed_data[column_names[0]] |
                            reversed_data[column_names[1]]),
        _cn[2:7] + [_cn[10]]] = 0

        return reversed_data

# TODO consolidate code
class MetaEmploymentNoSecondJob(CustomConstraint):
    @staticmethod
    def is_valid(column_names, data):
        return validate(column_names, data, False)

    @staticmethod
    def transform(column_names: tuple[str, ...] | list[str, ...],
                  data: pd.DataFrame) -> pd.DataFrame:
        """Replaces structural zeroes in income and time with corresponding medians"""
        # TODO investigate full vs partial median

        _parameter_check(column_names, TARGET_ORDER[:-1])

        _cn = list(column_names)

        transformed_data = data.copy()

        # self-employed
        transformed_data = filler(transformed_data, _cn[2:5],
                                  transformed_data[column_names[0]])

        # employed
        transformed_data = filler(transformed_data, _cn[5:7],
                                  transformed_data[column_names[1]])

        # unemployed
        transformed_data = filler(transformed_data, _cn[2:7],
                                  ~(transformed_data[column_names[0]] |
                                    transformed_data[column_names[1]]))

        return transformed_data

    @staticmethod
    def reverse_transform(column_names: tuple[str, ...] | list[str, ...],
                          data: pd.DataFrame) -> pd.DataFrame:
        """Does the inverse transformation,
         pads corresponding time and income columns for non-(self)employed
         with zeroes"""

        _parameter_check(column_names, TARGET_ORDER[:-1])

        _cn = list(column_names)

        reversed_data = data.copy()

        # self-employed
        reversed_data.loc[reversed_data[column_names[0]], _cn[2:5]] = 0

        # employed
        reversed_data.loc[reversed_data[column_names[1]], _cn[5:7]] = 0

        # unemployed
        reversed_data.loc[~(reversed_data[column_names[0]] |
                            reversed_data[column_names[1]]), _cn[2:7]] = 0

        return reversed_data

class BenefitsIncome(CustomConstraint):
    @staticmethod
    def is_valid(column_names, data):
        # any benefit means non-zero income
        _mask = data[column_names[1:]].any(axis=1)
        return (data[_mask][column_names[0]].gt(0) |
                data[~_mask][column_names[0]].eq(0))

    @staticmethod
    def transform(column_names, data):
        """Replaces structural zeroes in income and time with corresponding medians"""
        transformed_data = data.copy()

        _mask = transformed_data[column_names[1:]].any(axis=1)
        typical_benefit = round(transformed_data[_mask][column_names[0]].median()) + 1

        transformed_data.loc[~_mask, column_names[0]] = typical_benefit
        return transformed_data

    @staticmethod
    def reverse_transform(column_names, data: pd.DataFrame):
        reversed_data = data.copy()

        _mask = reversed_data[column_names[1:]].any(axis=1)

        reversed_data.loc[~_mask, column_names[0]] = 0
        return reversed_data
