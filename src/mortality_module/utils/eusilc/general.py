import pandas as pd

def typer(df_: pd.DataFrame) -> pd.DataFrame:
    _map = dict()
    for _column in df_.columns:
        if "indicator_" in _column:
            _map[_column] = "bool[pyarrow]"
        if "ordinal_" in _column:
            if _column not in ["ordinal_age_band"]:
                _map[_column] = "uint8[pyarrow]"
                # TODO deal with age bands
        if "category_" in _column:
            _map[_column] = "uint16[pyarrow]"
    return df_.astype(_map)

def get_working_status(_record):
    if pd.isna(_record):
        return pd.NA, pd.NA
    match _record:
        case 20: return True, False # employed, not self-employed
        case 10: return False, True # not employed, self-employed
        case -2: return False, False
        case _: return pd.NA, pd.NA

def get_full_time_status(_record):
    if pd.isna(_record):
        return pd.NA
    match _record:
        case 1: return True
        case 2: return False
        case -2: return False
        case -1: return pd.NA
        case _: return pd.NA

def get_birth_location(_record):
    match _record:
        case 10: return True
        case 21: return False
        case 22: return False
        case _: return pd.NA

def _age_min_max(_x):
    match _x:
        case '15-17': return 15, 17
        case '18-19': return 18, 19
        case '20-24': return 20, 24
        case '25-29': return 25, 29
        case '30-34': return 30, 34
        case '35-39': return 35, 39
        case '40-44': return 40, 44
        case '45-49': return 45, 49
        case '50-54': return 50, 54
        case '55-59': return 55, 59
        case '60-64': return 60, 64
        case '65-69': return 65, 69
        case '70-74': return 70, 74
        case '75-79': return 75, 79
        case '80-84': return 80, 84
        case _: return 85, 120
    # NOTE: the last upper limit should be infinity, but we pick a more practical value

def _get_ci(_x):
    _t = _x["ordinal_bmi_height"] * 0.01
    return _x["ordinal_bmi_weight"] / (_t * _t * _t)

def _health_band(_ci: float):
    if _ci < 11: return 0
    if _ci < 15: return 1
    if _ci < 17: return 2
    return 3

def _age_band(_x):
    if _x.ordinal_age_band_x == _x.ordinal_age_band_y:
        return 0
    else:
        if ((_x.ordinal_age == _x.age_band_max_x) or
            (_x.ordinal_age == _x.age_band_min_x)) and ((abs(_x.ordinal_age - _x.age_band_max_y) == 1) or
                                                 (abs(_x.ordinal_age - _x.age_band_min_y) == 1)):
            return 1
    return 100 # penalty
# FIXME we assume no na here

def _ordinal(_x, column_name, deviation_limit=2):
    d = abs(_x[column_name + "_x"] - _x[column_name + "_y"])
    return 100 if d > deviation_limit else d
    # penalty again
