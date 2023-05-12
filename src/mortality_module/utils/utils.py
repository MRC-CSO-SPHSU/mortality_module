from pathlib import Path
from typing import Any


def path_validation(path_: Any) -> Path:
    """Ensures that the argument is a valid Path object or string.

    Runs a series of checks to make sure there is no problems with a file path.

    Parameters
    ----------
        path_ : str
            The path to a directory.

    Returns
    -------
    Path
        The input Path, or the input string converted to one.

    Raises
    ------
    TypeError
        When the input is `None`, or not a string/Path object.
    ValueError
        When the Path is not a directory.

    """
    if path_ is None:
        raise TypeError('Provided path is None.')

    if not (isinstance(path_, str) or isinstance(path_, Path)):
        raise TypeError(f'Provided path is not a string/Path object but'
                        f' {type(path_)}.')

    p_ = path_ if isinstance(path_, Path) else Path(path_)

    if not p_.is_dir():
        raise ValueError('Provided Path is not a directory.')

    return p_
