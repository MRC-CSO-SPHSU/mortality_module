def get_patterns(_df, column_list):
    _mask = _df[column_list].isna().drop_duplicates()
    _mask = _mask[_mask.any(axis=1)]
    return _mask.reset_index(drop=True)

def correct_imputed_data(_df):
    pass
