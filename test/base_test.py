import unittest
import sys
import importlib
from importlib import reload
sys.path.append('./')
import numpy as np
import logging
from inference import *


class BasePythonTest(unittest.TestCase):
    cnt = 0

    def __init__(self, *args, **kwargs):
        super(BasePythonTest, self).__init__(*args, **kwargs)

    def assertEqualOperator(self, lhs, rhs):
        assert isinstance(lhs, TypeOperator) and isinstance(rhs, TypeOperator)
        assert lhs.name == rhs.name
        assert len(lhs.types) == len(rhs.types)
        if len(lhs.types) > 0:
            for ty_index in range(len(lhs.types)):
                self.assertEqualOperator(lhs.types[ty_index], rhs.types[ty_index])

    def test_1(self):
        var1 = TypeVariable()
        my_env = {"true": Bool,
                  "test_f": generalize({}, Function(var1, var1)),
                  "merge": Function(Integer, Function(Bool, Bool))}
        example = Apply(Apply(Identifier("merge"), Apply(Identifier("test_f"), Identifier("3"))), Apply(Identifier("test_f"), Identifier("true")))
        res = try_exp(my_env, example)
        self.assertEqualOperator(res, Bool)