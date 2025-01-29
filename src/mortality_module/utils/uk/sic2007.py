import pandas as pd


def map_sic2007(_df: pd.DataFrame) -> pd.DataFrame:
    # must have Class and Division columns

    _df.loc[_df["Division"].isin([1, 2]), "USoc"] = 1
    _df.loc[_df["Division"].isin([3]), "USoc"] = 2
    _df.loc[_df["Division"].isin([35, 36, 6, 23, 37]), "USoc"] = 3
    # _df.loc[_df["Division"].isin([5, 7, 8, 9]) | (_df["Class"].ge(19200) & _df["Class"].lt(19300)), "USoc"] = 4
    _df.loc[_df["Division"].isin([5, 7, 8, 9]) |
            (_df["Class"].ge(19100) & _df["Class"].lt(19300)), "USoc"] = 4
    _df.loc[_df["Division"].isin([20, 21]), "USoc"] = 5
    _df.loc[_df["Class"].ge(22100) & _df["Class"].lt(22300), "USoc"] = 6
    _df.loc[_df["Division"].isin([23]), "USoc"] = 7
    _df.loc[_df["Division"].isin([24]), "USoc"] = 8
    _df.loc[_df["Division"].isin([25, 28, 29, 30]) |
           (_df["Class"].ge(33110) & _df["Class"].lt(33130)) |
           (_df["Class"].ge(33200) & _df["Class"].lt(33300)) |
           (_df["Class"].ge(45200) & _df["Class"].lt(45300)) |
           _df["Class"].eq(27520), "USoc"] = 9
    _df.loc[_df["Division"].isin([26]) |
           (_df["Class"].ge(33130) & _df["Class"].lt(33150)) |
           (_df["Class"].ge(95100) & _df["Class"].lt(95200)) |
           (_df["Class"].ge(95210) & _df["Class"].lt(95220)) |
           (_df["Division"].eq(27) & _df["Class"].ne(27520)), "USoc"] = 10
    _df.loc[_df["Division"].isin([16, 17, 18, 31, 58]) |
           (_df["Class"].ge(95240) & _df["Class"].lt(95250)), "USoc"] = 11
    _df.loc[_df["Division"].isin([13, 14, 15, 32]) |
           (_df["Class"].ge(33190) & _df["Class"].lt(33200)) |
           (_df["Class"].ge(95220) & _df["Class"].lt(95240)) |
           (_df["Class"].ge(95250) & _df["Class"].lt(95260)) |
           (_df["Class"].ge(95290) & _df["Class"].lt(95300)) |
           (_df["Class"].ge(74100) & _df["Class"].lt(74200)), "USoc"] = 12
    _df.loc[_df["Division"].isin([10, 11, 12]), "USoc"] = 13
    _df.loc[_df["Division"].isin([41, 42]), "USoc"] = 14
    _df.loc[_df["Division"].isin([43]), "USoc"] = 15
    _df.loc[(_df["Class"].ge(46200) & _df["Class"].lt(47000)) |
            _df["Class"].isin([45310]), "USoc"] = 16
    _df.loc[_df["Class"].ge(46100) & _df["Class"].lt(46200), "USoc"] = 17
    _df.loc[_df["Division"].isin([47]) |
           (_df["Class"].ge(45100) & _df["Class"].lt(45200)) |
           (_df["Class"].ge(45311) & _df["Class"].lt(45500)), "USoc"] = 18
    _df.loc[(_df["Class"].ge(49100) & _df["Class"].lt(49300)) |
            _df["Class"].isin([49311, 52211, 52212]), "USoc"] = 19
    _df.loc[_df["Division"].isin([53, 60, 61, 90, 91]) |
           (_df["Class"].ge(74200) & _df["Class"].lt(74300)) |
           (_df["Class"].ge(59100) & _df["Class"].lt(59300)), "USoc"] = 20
    _df.loc[_df["Division"].isin([50, 51, 79]) |
           (_df["Class"].ge(33150) & _df["Class"].lt(33180)) |
           (_df["Class"].ge(49300) &
            _df["Class"].lt(49600) &
            _df["Class"].ne(49311)) |
           (_df["Division"].isin([52]) &
            _df["Class"].ne(52211) &
            _df["Class"].ne(52212)), "USoc"] = 21
    _df.loc[_df["Division"].isin([64]) |
           (_df["Class"].ge(66100) & _df["Class"].lt(66200)) |
           (_df["Class"].ge(66300) & _df["Class"].lt(66400)) |
           (_df["Class"].ge(69100) & _df["Class"].lt(69200)) |
           _df["Class"].eq(70221), "USoc"] = 22
    _df.loc[_df["Division"].isin([65]) |
            (_df["Class"].ge(66200) & _df["Class"].lt(66300)), "USoc"] = 23
    _df.loc[_df["Division"].isin([55, 56]), "USoc"] = 24
    _df.loc[_df["Division"].isin([96]), "USoc"] = 25
    _df.loc[_df["Division"].isin([38, 39]), "USoc"] = 26
    _df.loc[_df["Division"].isin([85, 93, 72, 92]), "USoc"] = 27
    _df.loc[_df["Division"].isin([86, 75]), "USoc"] = 28
    _df.loc[_df["Division"].isin([68]) |
            (_df["Class"].ge(69100) & _df["Class"].lt(69300)), "USoc"] = 29
    _df.loc[_df["Division"].isin([62, 63, 71, 80, 81, 82, 77, 78]) |
           (_df["Class"].ge(74300) & _df["Class"].lt(74400)) |
           (_df["Class"].ge(73100) & _df["Class"].lt(73300)) |
           (_df["Class"].ge(74900) & _df["Class"].lt(75000)) |
           (_df["Division"].eq(70) & _df["Class"].ne(70221)), "USoc"] = 30
    _df.loc[_df["Division"].isin([87, 88, 94]), "USoc"] = 31
    _df.loc[_df["Division"].isin([97, 98]), "USoc"] = 32
    _df.loc[_df["Division"].isin([99]) |
            (_df["Class"].ge(84100) & _df["Class"].lt(84300)), "USoc"] = 33
    _df.loc[_df["Class"].ge(84300) & _df["Class"].lt(84400), "USoc"] = 34
    return _df
