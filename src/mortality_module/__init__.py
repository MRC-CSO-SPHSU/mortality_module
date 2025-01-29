from importlib.metadata import version

__version__ = version("mortality_module")

from mortality_module.utils.data_utils import load_census_data
from mortality_module.utils.data_utils import load_mortality_rates
from mortality_module.utils.data_utils import load_birth_numbers

from mortality_module.dynamics.ageing import Ageing
