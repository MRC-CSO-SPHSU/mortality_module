UK_SEX_MAP = {1: 'm',
              2: 'f'}

UK_COUNTRY_MAP = {1: 'e',
                  2: 'ni',
                  3: 's',
                  4: 'w'}

UK_HH_CODES = {
1: "1 person", # a
2: "2 or more persons, all different family units", # b
3: "Married couple, no children, no other family units", # c
4: "Cohabiting couple, no children, no other family units", # c
5: "Couple, no children, other family units", # d
6: "Married couple, all dependent children, no other family units",  # e
7: "Cohabiting couple, all dependent children, no other family units", # e
8: "Married couple, dependent & non-dependent children, no other family units", # e
9: "Cohabiting couple, dependent & non dependent children, no other family units", # e
10: "Married couple, all non-dependent children, no other family units", # e
11: "Cohabiting couple, all non-dependent children, no other family units", # e
12: "Couple, all dependent children, other family units", # f
13: "Couple, dependent & non-dependent children, other family units", # f
14: "Couple, all non-dependent children, other family units", # f
15: "Lone parent, all dependent children, no other family units", # g
16: "Lone parent, dependent & non-dependent children, no other family units", # g
17: "Lone parent, all non-dependent children, no other family units", # g
18: "Lone parent, all dependent children, other family units", # h
19: "Lone parent, dependent & non-dependent children, other family units", #h
20: "Lone parent, all non-dependent children, other family units", # h
21: "2 or more family units, all dependent children", # i
22: "2 or more family units, dependent & non-dependent children", # i
23: "2 or more family units, all non-dependent children", # i
24: "2 or more family units, no children", # j
25: "Same sex couple with or without others", # no
26: "Civil partners/same sex marriage, with or without others (from July 2014)" # no
}

UK_NEW_HH_CODES = {
    'a': [1],
    'b': [2],
    'c': [3, 4],
    'd': [5],
    'e': [6, 7, 8, 9, 10, 11],
    'f': [12, 13, 14],
    'g': [15, 16, 17],
    'h': [18,19, 20],
    'i': [21, 22, 23],
    'j': [24]
}

RELFHU_VALUES = {
    1: 'Head of family',
    2: 'Wife/partner of head of family',
    3: 'Child of head of family/other person'
}

CAIND_VALUES = {
    1: 'Adult',
    2: 'Child of head of household and head of family unit',
    3: 'Child of other family',
    4: 'Child of head of household but not head of family unit'
}
