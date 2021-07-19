import unittest
import sys
import importlib
from importlib import reload
sys.path.append('./')
import numpy as np
import logging
from inference import *
from inference_logger import set_level


class BasePythonTest(unittest.TestCase):
    cnt = 0

    def __init__(self, *args, **kwargs):
        super(BasePythonTest, self).__init__(*args, **kwargs)
        set_level(logging.ERROR)

    def assertEqualOperator(self, lhs, rhs):
        assert isinstance(lhs, TypeOperator) and isinstance(rhs, TypeOperator)
        assert lhs.name == rhs.name
        assert len(lhs.types) == len(rhs.types)
        if len(lhs.types) > 0:
            for ty_index in range(len(lhs.types)):
                self.assertEqualOperator(lhs.types[ty_index], rhs.types[ty_index])
                
    def test_inference1(self):
        my_env = {"f": Function(Integer, Bool)}
        example = Apply(Identifier("f"), Identifier("3"))
        ty, mgu = infer_exp(my_env, example)
        self.assertEqualOperator(ty, Bool)

    def test_inference2(self):
        var1 = TypeVariable()
        my_env = {"true": Bool,
                  "add": Function(Integer, Function(Integer, Integer)),
                  "test_f": generalize({}, Function(var1, var1)),
                  "merge": Function(Integer, Function(Bool, Bool)),
                  "z": TypeVariable()}
        example = Lambda("x", Lambda("y", Apply(Apply(Identifier("add"), Identifier("z")), Identifier("3"))))
        ty, mgu = infer_exp(my_env, example)
        z_ty = apply(mgu, my_env['z'])
        self.assertEqualOperator(z_ty, Integer)

    def test_generalization(self):
        var1 = TypeVariable()
        my_env = {"true": Bool,
                  "test_f": generalize({}, Function(var1, var1)),
                  "merge": Function(Integer, Function(Bool, Bool))}
        example = Apply(Identifier("test_f"), Identifier("3"))
        res, mgu = infer_exp(my_env, example)
        self.assertEqualOperator(res, Integer)
        #
        example = Apply(Apply(Identifier("merge"), Apply(Identifier("test_f"), Identifier("3"))), Apply(Identifier("test_f"), Identifier("true")))
        res, mgu = infer_exp(my_env, example)
        self.assertEqualOperator(res, Bool)
        #
        example = Let("f", Lambda("x", Identifier("x")),
            Apply(Apply(Identifier("merge"), Apply(Identifier("f"), Identifier("3"))),
                  Apply(Identifier("f"), Identifier("true"))))
        res, mgu = infer_exp(my_env, example)
        self.assertEqualOperator(res, Bool)