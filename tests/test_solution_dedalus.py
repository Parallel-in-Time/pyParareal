import sys
sys.path.append('./src')
import dedalus.public as d3

from solution import solution
from solution_linear import solution_linear
from solution_dedalus import solution_dedalus
import numpy as np
import pytest

class TestClass:

    # Can instantiate
    def test_caninstantiate(self):     
        ndof = 32
        sol = solution_dedalus(np.zeros(ndof), ndof)
