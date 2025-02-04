from importlib.metadata import version

__version__ = version("synthwave")

from synthwave.utils.data_utils import load_census_data
from synthwave.utils.data_utils import load_mortality_rates
from synthwave.utils.data_utils import load_birth_numbers

from synthwave.dynamics.ageing import Ageing
