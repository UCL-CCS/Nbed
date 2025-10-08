"""Types and type alisases used in Nbed."""

from typing import Literal

import numpy as np

type OneSpinMatrix[M: int] = np.ndarray[tuple[M, M], np.dtype[np.floating]]
type TwoSpinMatrix[M: int] = np.ndarray[tuple[Literal[2], M, M], np.dtype[np.floating]]
