from typing import Any

import pandas as pd
from abc import ABC, abstractmethod
# TODO make sure validators work fine with regular OR and don't actually need XOR

class CustomConstraint(ABC):
    @staticmethod
    @abstractmethod
    def is_valid(column_names: tuple[str, ...] | list[str, ...],
                 data: pd.DataFrame) -> pd.Series:
        """Tests if all rows are valid. Must return all True"""
        pass

    @staticmethod
    @abstractmethod
    def transform(column_names: tuple[str, ...] | list[str, ...],
                  data: pd.DataFrame) -> pd.DataFrame:
        pass

    @staticmethod
    @abstractmethod
    def reverse_transform(column_names: tuple[str, ...] | list[str, ...],
                          data: pd.DataFrame) -> pd.DataFrame:
        pass

    @classmethod
    def get_schema(cls, _columns) -> dict:
        return {
                'constraint_class': cls.__name__,
                'constraint_parameters': {
                    'column_names': _columns,
                }
            }

