#!/usr/bin/env python
"""
.. module:: inference
   :synopsis: An implementation of the Hindley Milner type checking algorithm
              based on the Scala code by Andrew Forrest, the Perl code by
              Nikita Borisov and the paper "Basic Polymorphic Typechecking"
              by Cardelli.
.. moduleauthor:: Robert Smallshire
"""

from __future__ import print_function
from enum import IntEnum
import copy


class ConsType(IntEnum):
    ConsInvalid = -1
    ConsEq = 0
    ConsLess = 1
    ConsLessM = 2


class TypeConstraint(object):
    def __init__(self, lhs=None, rhs=None, ctype=ConsType.ConsInvalid, mid=None):
        self.lhs = lhs
        self.rhs = rhs
        self.mid = mid
        self.ctype = ctype
        if ctype == ConsType.ConsLess and mid is not None:
            self.ctype = ConsType.ConsLessM

    def __str__(self):
        return "type:{}, lhs:{}, rhs:{}, mid:{}".format(self.ctype, self.lhs, self.rhs, self.mid)

    def get_active_vars(self):
        if self.ctype == ConsType.ConsLessM:
            return free_type_variable(self.lhs).union(free_type_variable(self.mid).intersection(free_type_variable(self.rhs)))
        return free_type_variable(self.lhs).union(free_type_variable(self.rhs))

# =======================================================#
# Class definitions for the abstract syntax tree nodes
# which comprise the little language for which types
# will be inferred

class AstTreeNode(object):
    def __init__(self):
        self.m_set = set()   # monomorphic set
        
    def add_m(self, tyv):
        self.m_set.add(tyv)


class Lambda(AstTreeNode):
    """Lambda abstraction"""
    def __init__(self, v, body):
        super().__init__()
        self.v = v
        self.body = body

    def __str__(self):
        return "(fn {v} => {body})".format(v=self.v, body=self.body)

    def add_m(self, tyv):
        super(Lambda, self).add_m(tyv)
        self.body.add_m(tyv)


class Identifier(AstTreeNode):
    """Identifier"""
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __str__(self):
        return self.name


class Apply(AstTreeNode):
    """Function application"""
    def __init__(self, fn, arg):
        super().__init__()
        self.fn = fn
        self.arg = arg

    def __str__(self):
        return "({fn} {arg})".format(fn=self.fn, arg=self.arg)

    def add_m(self, tyv):
        super(Apply, self).add_m(tyv)
        self.fn.add_m(tyv)
        self.arg.add_m(tyv)


class Let(AstTreeNode):
    """Let binding"""
    def __init__(self, v, defn, body):
        super().__init__()
        self.v = v
        self.defn = defn
        self.body = body

    def __str__(self):
        return "(let {v} = {defn} in {body})".format(v=self.v, defn=self.defn, body=self.body)

    def add_m(self, tyv):
        super(Let, self).add_m(tyv)
        self.defn.add_m(tyv)
        self.body.add_m(tyv)


class Letrec(AstTreeNode):
    """Letrec binding"""
    def __init__(self, v, defn, body):
        super().__init__()
        self.v = v
        self.defn = defn
        self.body = body

    def __str__(self):
        return "(letrec {v} = {defn} in {body})".format(v=self.v, defn=self.defn, body=self.body)

    def add_m(self, tyv):
        super(Letrec, self).add_m(tyv)
        self.defn.add_m(tyv)
        self.body.add_m(tyv)

# =======================================================#
# Exception types

class InferenceError(Exception):
    """Raised if the type inference algorithm cannot infer types successfully"""

    def __init__(self, message):
        self.__message = message

    message = property(lambda self: self.__message)

    def __str__(self):
        return str(self.message)


class ParseError(Exception):
    """Raised if the type environment supplied for is incomplete"""

    def __init__(self, message):
        self.__message = message

    message = property(lambda self: self.__message)

    def __str__(self):
        return str(self.message)


# =======================================================#
# Types and type constructors

class TypeVariable(object):
    """A type variable standing for an arbitrary type.

    All type variables have a unique id, but names are only assigned lazily,
    when required.
    """

    next_variable_id = 0

    def __init__(self):
        self.id = TypeVariable.next_variable_id
        TypeVariable.next_variable_id += 1
        self.instance = None
        self.__name = None

    next_variable_name = 'a'

    @property
    def name(self):
        """Names are allocated to TypeVariables lazily, so that only TypeVariables
        present after analysis consume names.
        """
        if self.__name is None:
            self.__name = TypeVariable.next_variable_name
            TypeVariable.next_variable_name = chr(ord(TypeVariable.next_variable_name) + 1)
        return self.__name

    def __str__(self):
        if self.instance is not None:
            return str(self.instance)
        else:
            return self.name

    def __repr__(self):
        return "TypeVariable(id = {0}, name={1})".format(self.id, self.name)


class TypeScheme(object):
    def __init__(self, name, quantified_types=[], type_op=None):
        self.name = name
        self.quantified_types = quantified_types
        self.type_op = type_op

    def instantiate(self):
        new_types = []
        for ty in self.type_op.types:
            if ty in self.quantified_types:
                new_types.append(TypeVariable())
            else:
                new_types.append(ty)
        return TypeOperator(self.name, new_types)

    def get_ftvs(self):
        ftvs = self.type_op.get_ftvs()
        return ftvs.difference(set(self.quantified_types))


class TypeOperator(object):
    """An n-ary type constructor which builds a new type from old"""
    def __init__(self, name, types):
        self.name = name
        self.types = types

    def __str__(self):
        num_types = len(self.types)
        if num_types == 0:
            return self.name
        elif num_types == 2:
            return "({0} {1} {2})".format(str(self.types[0]), self.name, str(self.types[1]))
        else:
            return "{0} {1}" .format(self.name, ' '.join(self.types))

    def __repr__(self):
        return self.__str__()

    def generalize(self, env):
        tyv_list = []
        for ty in self.types:
            if isinstance(ty, TypeVariable) and ty not in env:
                tyv_list.append(ty)
        type_scheme = TypeScheme(self.name, tyv_list, self)
        return type_scheme

    def get_ftvs(self):
        ftvs = set()
        for ty in self.types:
            if isinstance(ty, TypeVariable):
                ftvs.add(ty)
        return ftvs


class Function(TypeOperator):
    """A binary type constructor which builds function types"""
    def __init__(self, from_type, to_type):
        super(Function, self).__init__("->", [from_type, to_type])


def free_type_variable(x):
    if isinstance(x, TypeScheme):
        return x.get_ftvs()
    elif isinstance(x, TypeOperator):
        return x.get_ftvs()
    elif isinstance(x, set) or isinstance(x, list):
        ftvs = set()
        for e in x:
            ftvs.union(free_type_variable(e))
        return ftvs
    return set()


def instantiate(x):
    if isinstance(x, TypeScheme):
        return x.instantiate()
    else:
        return x

def get_active_vars(cons):
    atvs = set()
    for c in cons:
        atvs.union(c.get_active_vars())
    return atvs

def ftv(x):
    if isinstance(x, Function):
        return ftv(x.types[0]) | ftv(x.types[1])
    elif isinstance(x, Apply):
        return ftv(x.fn) | ftv(x.arg)
    elif isinstance(x, Let):
        return ftv(x.body).set_A.difference(ftv(x.v))
    # elif isinstance(x, TypeOperator):
    #     return ftv(x.fn) | ftv(x.arg)
    elif isinstance(x, TypeVariable):
        return set([x])
    else:
        return set()


# Basic types are constructed with a nullary type constructor
Integer = TypeOperator("int", [])  # Basic integer
Bool = TypeOperator("bool", [])  # Basic bool


# =======================================================#
# Type inference machinery
constraint = []
def analyse(node, env=None):
    """Computes the type of the expression given by node.

    The type of the node is computed in the context of the
    supplied type environment env. Data types can be introduced into the
    language simply by having a predefined set of identifiers in the initial
    environment. environment; this way there is no need to change the syntax or, more
    importantly, the type-checking program when extending the language.

    Args:
        node: The root of the abstract syntax tree.
        env: The type environment is a mapping of expression identifier names
            to type assignments.
            to type assignments.
        non_generic: A set of non-generic variables, or None

    Returns:
        The computed type of the expression.

    Raises:
        InferenceError: The type of the expression could not be inferred, for example
            if it is not possible to unify two types such as Integer and Bool
        ParseError: The abstract syntax tree rooted at node could not be parsed
    """
    assum = {}     # assumptions
    cons = set()   # constraints
    if isinstance(node, Identifier):
        if node.name in env:
            v_type = env[node.name]
        elif is_integer_literal(node.name):
            v_type = Integer
        else:
            v_type = TypeVariable()
            assum[node.name] = v_type
        return v_type, assum, cons
    elif isinstance(node, Apply):
        fun_type, assum, cons1 = analyse(node.fn, env)
        arg_type, assum2, cons2 = analyse(node.arg, env)
        result_type = TypeVariable()
        assum.update(assum2)
        cons = cons1.union(cons2)
        cons.add(TypeConstraint(Function(arg_type, result_type), fun_type, ConsType.ConsEq))
        return result_type, assum, cons
    elif isinstance(node, Lambda):
        arg_type = TypeVariable()
        node.body.add_m(arg_type)
        result_type, assum, cons = analyse(node.body, env)
        x_type = assum[node.v]
        del assum[node.v]
        cons.add(TypeConstraint(x_type, arg_type, ConsType.ConsEq))
        return Function(arg_type, result_type), assum, cons
    elif isinstance(node, Let):
        defn_type, assum, cons1 = analyse(node.defn, env)
        body_type, assum2, cons2 = analyse(node.body, env)
        x_type = assum2[node.v]
        assum.update(assum2)
        del assum[node.v]
        cons = cons1.union(cons2)
        cons.add(TypeConstraint(x_type, defn_type, ConsType.ConsLess, node.m_set))
        return body_type, assum, cons
    elif isinstance(node, Letrec):
        new_type = TypeVariable()
        new_env = env.copy()
        new_env[node.v] = new_type
        new_non_generic = non_generic.copy()
        new_non_generic.add(new_type)
        defn_type = analyse(node.defn, new_env, new_non_generic)
        # unify(new_type, defn_type)
        constraint.append(TypeConstraint(new_type, defn_type, ConsType.ConsEq))
        return analyse(node.body, new_env, non_generic), assum, cons
    assert 0, "Unhandled syntax node {0}".format(type(node))


def get_type(name, env, non_generic):
    """Get the type of identifier name from the type environment env.

    Args:
        name: The identifier name
        env: The type environment mapping from identifier names to types
        non_generic: A set of non-generic TypeVariables

    Raises:
        ParseError: Raised if name is an undefined symbol in the type
            environment.
    """
    if name in env:
        return fresh(env[name], non_generic)
    elif is_integer_literal(name):
        return Integer
    else:
        raise ParseError("Undefined symbol {0}".format(name))


def fresh(t, non_generic):
    """Makes a copy of a type expression.

    The type t is copied. The the generic variables are duplicated and the
    non_generic variables are shared.

    Args:
        t: A type to be copied.
        non_generic: A set of non-generic TypeVariables
    """
    mappings = {}  # A mapping of TypeVariables to TypeVariables

    def freshrec(tp):
        p = prune(tp)
        if isinstance(p, TypeVariable):
            if is_generic(p, non_generic):
                if p not in mappings:
                    mappings[p] = TypeVariable()
                return mappings[p]
            else:
                return p
        elif isinstance(p, TypeOperator):
            return TypeOperator(p.name, [freshrec(x) for x in p.types])

    return freshrec(t)


# def unify(t1, t2):
#     """Unify the two types t1 and t2.
#
#     Makes the types t1 and t2 the same.
#
#     Args:
#         t1: The first type to be made equivalent
#         t2: The second type to be be equivalent
#
#     Returns:
#         None
#
#     Raises:
#         InferenceError: Raised if the types cannot be unified.
#     """
#
#     a = prune(t1)
#     b = prune(t2)
#     if isinstance(a, TypeVariable):
#         if a != b:
#             if occurs_in_type(a, b):
#                 raise InferenceError("recursive unification")
#             a.instance = b
#     elif isinstance(a, TypeOperator) and isinstance(b, TypeVariable):
#         unify(b, a)
#     elif isinstance(a, TypeOperator) and isinstance(b, TypeOperator):
#         if a.name != b.name or len(a.types) != len(b.types):
#             raise InferenceError("Type mismatch: {0} != {1}".format(str(a), str(b)))
#         for p, q in zip(a.types, b.types):
#             unify(p, q)
#     else:
#         assert 0, "Not unified"

### == Constraint Solver ==
from collections import deque
def empty():
    return {}

def const_type(x):
    if isinstance(x, TypeOperator) and len(x.types) == 0:
        return True
    return False

def apply(s, t):
    if const_type(t) or isinstance(t, Identifier):
        return t
    elif isinstance(t, Apply):
        return Apply(apply(s, t.fn), apply(s, t.arg))
    elif isinstance(t, Let):
        return Let(t.v, apply(s, t.defn), apply(s, t.body))
    elif isinstance(t, Letrec):
        return Letrec(t.v, apply(s, t.defn), apply(s, t.body))
    elif isinstance(t, Lambda):
        return Lambda(t.v, apply(s, t.body))
    elif isinstance(t, Function):
        return Function(apply(s, t.types[0]), apply(s, t.types[1]))
    elif isinstance(t, TypeVariable):
        return s.get(t.name, t)
    elif isinstance(t, set):
        new_set = set()
        for item in t:
            new_set.add(apply(s, item))
        return new_set

def retrieve_type(node):
    if const_type(node) or isinstance(node, Identifier):
        return node
    elif isinstance(node, Apply):
        return retrieve_type(node.fn)
    elif isinstance(node, Let):
        return retrieve_type(node.body)
    elif isinstance(node, Letrec):
        return retrieve_type(node.body)
    elif isinstance(node, Lambda):
        return node
    elif isinstance(node, Function):
        return node.types[-1]
    elif isinstance(node, TypeOperator):
        return node
    elif isinstance(node, TypeVariable):
        return node

def applyList(s, xs):
    res_list = []
    for cons in xs:
        if cons.ctype <= ConsType.ConsLess:
            res_list.append(TypeConstraint(apply(s, cons.lhs), apply(s, cons.rhs), cons.ctype))
        else:
            res_list.append(TypeConstraint(apply(s, cons.lhs), apply(s, cons.rhs), cons.ctype, apply(s, cons.mid)))
    return res_list if isinstance(xs, list) else set(res_list)

class InferError(Exception):
    def __init__(self, ty1, ty2):
        self.ty1 = ty1
        self.ty2 = ty2

    def __str__(self):
        return '\n'.join([
            "Type mismatch: ",
            "Given: ", "\t" + str(self.ty1),
            "Expected: ", "\t" + str(self.ty2)
        ])

def unify(x, y):
    if isinstance(x, Apply) and isinstance(y, Apply):
        s1 = unify(x.fn, y.fn)
        s2 = unify(apply(s1, x.arg), apply(s1, y.arg))
        return compose(s2, s1)
    elif const_type(x) and const_type(y) and (x.name == y.name):
        return empty()
    elif isinstance(x, TypeOperator) and isinstance(y, TypeOperator):
        if len(x.types) != len(y.types):
            return Exception("Wrong number of arguments")
        # s1 = solve(zip([x.types[0]], [y.types[0]]))
        s1 = solve([TypeConstraint(x.types[0], x.types[1], ConsType.ConsEq)])
        s2 = unify(apply(s1, x.types[1]), apply(s1, y.types[1]))
        return compose(s2, s1)
    elif isinstance(x, TypeVariable):
        return bind(x.name, y)
    elif isinstance(y, TypeVariable):
        return bind(y.name, x)
    else:
        raise InferError(x, y)

def solve(xs):
    mgu = empty()
    cs = deque(xs)
    while len(cs):
        cons = cs.pop()
        s = unify(cons.lhs, cons.rhs)
        mgu = compose(s, mgu)
        cs = deque(applyList(s, cs))
    return mgu


def split_cons(cons):
    if len(cons) <= 1:
        return list(cons)[0], set()
    else:
        next_le_cons = None
        next_m_cons = None
        for con in cons:
            if con.ctype == ConsType.ConsEq:
                cons.remove(con)
                return con, cons
            elif con.ctype == ConsType.ConsLess:
                next_le_cons = con if next_le_cons is None else next_le_cons
            else:
                next_m_cons = con if next_m_cons is None else next_m_cons
        next_con = next_m_cons if next_m_cons is not None else next_le_cons
        cons.remove(next_con)
        return next_con, cons

def generalize(env, x):
    if isinstance(x, TypeOperator):
        return x.generalize(env)
    else:
        return x


def solve_cons(cons):
    # print("cur cons:{}".format(len(cons)))
    if len(cons) == 0:
        return dict()
    else:
        next_con, remain_cons = split_cons(cons)
        # print("next_con: {}".format(next_con))
        if next_con.ctype == ConsType.ConsEq:
            mgu = unify(next_con.lhs, next_con.rhs)
            return compose(solve_cons(applyList(mgu, remain_cons)), mgu)
        elif next_con.ctype == ConsType.ConsLessM:
            u = free_type_variable(next_con.rhs).difference(next_con.mid).intersection(get_active_vars(remain_cons))
            if u is None or len(u) == 0:
                new_cons = TypeConstraint(next_con.lhs, generalize(next_con.mid, next_con.rhs), ConsType.ConsLess)
                remain_cons.add(new_cons)
                return solve_cons(remain_cons)
        else:
            new_cons = TypeConstraint(next_con.lhs, instantiate(next_con.rhs), ConsType.ConsEq)
            remain_cons.add(new_cons)
            return solve_cons(remain_cons)


def bind(n, x):
    if x == n:
        return empty()
    elif occurs_check(n, x):
        raise InfiniteType(n, x)
    else:
        return dict([(n, x)])

def occurs_check(n, x):
    return n in ftv(x)

def union(s1, s2):
    nenv = s1.copy()
    nenv.update(s2)
    return nenv

def compose(s1, s2):
    s3 = dict((t, apply(s1, u)) for t, u in s2.items())
    return union(s1, s3)

def prune(t):
    """Returns the currently defining instance of t.

    As a side effect, collapses the list of type instances. The function Prune
    is used whenever a type expression has to be inspected: it will always
    return a type expression which is either an uninstantiated type variable or
    a type operator; i.e. it will skip instantiated variables, and will
    actually prune them from expressions to remove long chains of instantiated
    variables.

    Args:
        t: The type to be pruned

    Returns:
        An uninstantiated TypeVariable or a TypeOperator
    """
    if isinstance(t, TypeVariable):
        if t.instance is not None:
            t.instance = prune(t.instance)
            return t.instance
    return t


def is_generic(v, non_generic):
    """Checks whether a given variable occurs in a list of non-generic variables

    Note that a variables in such a list may be instantiated to a type term,
    in which case the variables contained in the type term are considered
    non-generic.

    Note: Must be called with v pre-pruned

    Args:
        v: The TypeVariable to be tested for genericity
        non_generic: A set of non-generic TypeVariables

    Returns:
        True if v is a generic variable, otherwise False
    """
    return not occurs_in(v, non_generic)


def occurs_in_type(v, type2):
    """Checks whether a type variable occurs in a type expression.

    Note: Must be called with v pre-pruned

    Args:
        v:  The TypeVariable to be tested for
        type2: The type in which to search

    Returns:
        True if v occurs in type2, otherwise False
    """
    pruned_type2 = prune(type2)
    if pruned_type2 == v:
        return True
    elif isinstance(pruned_type2, TypeOperator):
        return occurs_in(v, pruned_type2.types)
    return False


def occurs_in(t, types):
    """Checks whether a types variable occurs in any other types.

    Args:
        t:  The TypeVariable to be tested for
        types: The sequence of types in which to search

    Returns:
        True if t occurs in any of types, otherwise False
    """
    return any(occurs_in_type(t, t2) for t2 in types)


def is_integer_literal(name):
    """Checks whether name is an integer literal string.

    Args:
        name: The identifier to check

    Returns:
        True if name is an integer literal, otherwise False
    """
    result = True
    try:
        int(name)
    except ValueError:
        result = False
    return result


# ==================================================================#
# Example code to exercise the above


def try_exp(env, node):
    """Try to evaluate a type printing the result or reporting errors.

    Args:
        env: The type environment in which to evaluate the expression.
        node: The root node of the abstract syntax tree of the expression.

    Returns:
        None
    """
    global constraint
    constraint = []
    print(str(node) + " : ", end=' ')
    try:
        t, assum, cons = analyse(node, env)
        print("\nassumption: {}".format(assum))
        print("\nTotal cons: {}".format(len(cons)))
        # print("\nTotal cons: {}".format(cons))
        for item in assum:
            if item not in env:
                raise Exception("Undefined variables exist")
            cons.add(TypeConstraint(env[assum], env[item], ConsType.ConsEq))
        for con in cons:
            print(con)
        print("str: {}\n".format(str(t)))
        mgu = solve(cons)
        mgu1 = solve_cons(cons)
        print("mgu: {}\n".format(mgu))
        print("mgu1: {}\n".format(mgu1))
        infer_ty = apply(mgu, node)
        print("infer_ty: {}\n".format(infer_ty))
        res = retrieve_type(infer_ty)
        print("res: {}\n".format(res))
    except (ParseError, InferenceError) as e:
        print(e)


def main():
    """The main example program.

    Sets up some predefined types using the type constructors TypeVariable,
    TypeOperator and Function.  Creates a list of example expressions to be
    evaluated. Evaluates the expressions, printing the type or errors arising
    from each.

    Returns:
        None
    """

    var1 = TypeVariable()
    var2 = TypeVariable()
    pair_type = TypeOperator("*", (var1, var2))

    var3 = TypeVariable()

    my_env = {"pair": Function(var1, Function(var2, pair_type)),
              "true": Bool,
              "cond": Function(Bool, Function(var3, Function(var3, var3))),
              "zero": Function(Integer, Bool),
              "pred": Function(Integer, Integer),
              "times": Function(Integer, Function(Integer, Integer))}

    pair = Apply(Apply(Identifier("pair"),
                       Apply(Identifier("f"),
                             Identifier("4"))),
                 Apply(Identifier("f"),
                       Identifier("true")))

    examples = [
        # factorial
        Letrec("factorial",  # letrec factorial =
               Lambda("n",  # fn n =>
                      Apply(
                          Apply(  # cond (zero n) 1
                              Apply(Identifier("cond"),  # cond (zero n)
                                    Apply(Identifier("zero"), Identifier("n"))),
                              Identifier("1")),
                          Apply(  # times n
                              Apply(Identifier("times"), Identifier("n")),
                              Apply(Identifier("factorial"),
                                    Apply(Identifier("pred"), Identifier("n")))
                          )
                      )
                      ),  # in
               Apply(Identifier("factorial"), Identifier("5"))
               ),

        # Should fail:
        # fn x => (pair(x(3) (x(true)))
        Lambda("x",
               Apply(
                   Apply(Identifier("pair"),
                         Apply(Identifier("x"), Identifier("3"))),
                   Apply(Identifier("x"), Identifier("true")))),

        # pair(f(3), f(true))
        Apply(
            Apply(Identifier("pair"), Apply(Identifier("f"), Identifier("4"))),
            Apply(Identifier("f"), Identifier("true"))),

        # let f = (fn x => x) in ((pair (f 4)) (f true))
        Let("f", Lambda("x", Identifier("x")), pair),

        # fn f => f f (fail)
        Lambda("f", Apply(Identifier("f"), Identifier("f"))),

        # let g = fn f => 5 in g g
        Let("g",
            Lambda("f", Identifier("5")),
            Apply(Identifier("g"), Identifier("g"))),

        # example that demonstrates generic and non-generic variables:
        # fn g => let f = fn x => g in pair (f 3, f true)
        Lambda("g",
               Let("f",
                   Lambda("x", Identifier("g")),
                   Apply(
                       Apply(Identifier("pair"),
                             Apply(Identifier("f"), Identifier("3"))
                             ),
                       Apply(Identifier("f"), Identifier("true"))))),

        # Function composition
        # fn f (fn g (fn arg (f g arg)))
        Lambda("f", Lambda("g", Lambda("arg", Apply(Identifier("g"), Apply(Identifier("f"), Identifier("arg"))))))
    ]

    # try_exp(my_env, Apply(Identifier("zero"), Identifier("4")))
    # try_exp(my_env, Let("f", Lambda("x", Identifier("x")), Apply(Identifier("f"), Identifier("4"))))
    # try_exp(my_env, Let("y", Identifier("m"), Let("x", Apply(Identifier("y"), Identifier("4")), Identifier("x"))))
    try_exp(my_env, Lambda("m", Let("y", Identifier("m"), Let("x", Apply(Identifier("y"), Identifier("4")), Identifier("x")))))
    # for example in examples:
    #     try_exp(my_env, example)


if __name__ == '__main__':
    main()
