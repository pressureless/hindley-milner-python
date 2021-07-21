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
from inference_logger import log_content, log_perm


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
            return union_set(free_type_variable(self.lhs), intersect(free_type_variable(self.mid), free_type_variable(self.rhs)))
        return union_set(free_type_variable(self.lhs), free_type_variable(self.rhs))

    def satisfied(self, remain_cons):
        u = intersect(difference(free_type_variable(self.rhs), self.mid), get_active_vars(remain_cons))
        # log_content("ftv:{}, next_con.mid:{}, get_active_vars(remain_cons):{}".format(free_type_variable(self.rhs), self.mid, get_active_vars(remain_cons)))
        if u is None or len(u) == 0:
            return True
        return False

# =======================================================#
# Class definitions for the abstract syntax tree nodes
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
            self.__name = '_' + TypeVariable.next_variable_name
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
    def __init__(self, name, quantified_types=[], type_op=None, env=None):
        self.name = name
        self.quantified_types = quantified_types
        self.type_op = type_op
        self.env = env

    def instantiate(self):
        new_types = copy.deepcopy(self.type_op.types)
        if len(self.quantified_types) > 0:
            ty_dict = {}
            for ty in self.quantified_types:
                new_type = TypeVariable()
                ty_dict[ty.id] = new_type
                log_content("Instantiate TypeScheme, scheme:({}), {} -> {}".format(str(self), ty, new_type))
            for index in range(len(new_types)):
                if isinstance(new_types[index], TypeOperator):
                    new_types[index].instantiate(ty_dict)
                else:
                    if new_types[index].id in ty_dict:
                        new_types[index] = ty_dict[new_types[index].id]
        return TypeOperator(self.name, new_types)

    def get_ftvs(self):
        ftvs = self.type_op.get_ftvs()
        return difference(ftvs, set(self.quantified_types))

    def empty_quantified(self):
        return len(self.quantified_types) == 0

    def __str__(self):
        quantified_dsp = [str(tp) for tp in self.quantified_types]
        return "name:{}, quantified:{}, type:{}" .format(self.name, ' '.join(quantified_dsp), str(self.type_op))

    def __repr__(self):
        return self.__str__()


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
        new_env = env.copy()
        tyv_set = set()
        ftvs = self.get_ftvs()
        for ty in ftvs:
            if ty.name not in new_env:
                tyv_set.add(ty)
        log_content("Generalize ftvs: {}".format(ftvs))
        log_content("Generalize tyv_set: {}".format(tyv_set))
        type_scheme = TypeScheme(self.name, list(tyv_set), self, new_env)
        log_content("Generalize result: {}".format(str(type_scheme)))
        # print(str(type_scheme))
        return type_scheme

    def instantiate(self, ty_dict):
        if len(self.types) > 0:
            def modify_types():
                for cur_index in range(len(self.types)):
                    if isinstance(self.types[cur_index], TypeOperator):
                        self.types[cur_index].instantiate(ty_dict)
                    else:
                        if self.types[cur_index].id in ty_dict:
                            self.types[cur_index] = ty_dict[self.types[cur_index].id]
            if isinstance(self.types, list):
                modify_types()
            elif isinstance(self.types, tuple):
                self.types = list(self.types)
                modify_types()
                self.types = tuple(self.types)

    def get_ftvs(self):
        ftv_dict = {}
        for ty in self.types:
            if isinstance(ty, TypeVariable):
                ftv_dict[ty.name] = ty
                # ftvs.add(ty)
            elif isinstance(ty, TypeOperator):
                # type operator
                new_dict = gen_env_dict(ty.get_ftvs())
                ftv_dict.update(new_dict)
                # ftvs = ftvs.union()
        ftvs = set(ftv_dict.values())
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
        atvs = union_set(atvs, c.get_active_vars())
    return atvs


# Basic types are constructed with a nullary type constructor
Integer = TypeOperator("int", [])  # Basic integer
Double = TypeOperator("double", [])  # Basic double
Bool = TypeOperator("bool", [])  # Basic bool


# =======================================================#
# Type inference machinery
constraint = []


def merge_assum(lhs, rhs):
    # log_content("merge_assum, lhs:{}, rhs:{}".format(lhs, rhs))
    same_keys = set(lhs.keys()).intersection(rhs.keys())
    same_dict = {}
    if len(same_keys) > 0:
        def get_list(value):
            if isinstance(value, list):
                return value
            else:
                return [value]
        for k in same_keys:
            same_dict[k] = get_list(lhs[k]) + get_list(rhs[k])
    lhs.update(rhs)
    lhs.update(same_dict)
    # log_content("merge_assum, result:{}".format(lhs))
    return lhs


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
            v_type = instantiate(env[node.name])
            log_content("Node Identifier, existed {}:{}".format(node.name, str(v_type)))
            # v_type = env[node.name]
        elif is_integer_literal(node.name):
            v_type = Integer
        else:
            v_type = TypeVariable()
            assum[node.name] = v_type
            log_content("Node Identifier, Name {}:{}".format(node.name, v_type))
        return v_type, assum, cons
    elif isinstance(node, Apply):
        fun_type, assum, cons1 = analyse(node.fn, env)
        arg_type, assum2, cons2 = analyse(node.arg, env)
        result_type = TypeVariable()
        log_content("Node Apply, fun_type:{}, result name: {}".format(fun_type.name, result_type))
        cons = cons1.union(cons2)
        assum = merge_assum(assum, assum2)  # update assum
        cons.add(TypeConstraint(Function(arg_type, result_type), fun_type, ConsType.ConsEq))
        return result_type, assum, cons
    elif isinstance(node, Lambda):
        arg_type = TypeVariable()
        node.body.add_m(arg_type)
        result_type, assum, cons = analyse(node.body, env)
        if node.v in assum:
            # parameter may not be used inside function
            x_type = assum[node.v]
            del assum[node.v]
            if isinstance(x_type, list):
                for x_tp in x_type:
                    cons.add(TypeConstraint(x_tp, arg_type, ConsType.ConsEq))
            else:
                cons.add(TypeConstraint(x_type, arg_type, ConsType.ConsEq))
        log_content("Node Lambda, arg name {}:{}".format(node.v, arg_type))
        return Function(arg_type, result_type), assum, cons
    elif isinstance(node, Let):
        defn_type, assum, cons1 = analyse(node.defn, env)
        body_type, assum2, cons2 = analyse(node.body, env)
        x_type = assum2[node.v]
        cons = cons1.union(cons2)
        assum = merge_assum(assum, assum2)  # update assum
        del assum[node.v]
        if isinstance(x_type, list):
            for x_tp in x_type:
                cons.add(TypeConstraint(x_tp, defn_type, ConsType.ConsLess, node.m_set))
        else:
            cons.add(TypeConstraint(x_type, defn_type, ConsType.ConsLess, node.m_set))
        return body_type, assum, cons
    elif isinstance(node, Letrec):
        defn_type, assum, cons1 = analyse(node.defn, env)
        body_type, assum2, cons2 = analyse(node.body, env)
        x_type = assum2[node.v]
        cons = cons1.union(cons2)
        assum = merge_assum(assum, assum2)  # update assum
        del assum[node.v]
        if isinstance(x_type, list):
            for x_tp in x_type:
                cons.add(TypeConstraint(x_tp, defn_type, ConsType.ConsLess, node.m_set))
        else:
            cons.add(TypeConstraint(x_type, defn_type, ConsType.ConsLess, node.m_set))
        return body_type, assum, cons
    assert 0, "Unhandled syntax node {0}".format(type(node))


### == Constraint Solver ==
def empty():
    return {}


def const_type(x):
    if isinstance(x, TypeOperator) and len(x.types) == 0:
        return True
    return False


def apply(s, t):
    if const_type(t) or isinstance(t, Identifier):
        return t
    elif isinstance(t, TypeOperator):
        def get_list():
            new_dict = {}
            res = []
            for ty in t.types:
                res.append(apply(s, ty))
            return res
        if isinstance(t.types, list):
            return TypeOperator(t.name, get_list())
        else:
            return TypeOperator(t.name, tuple(get_list()))
    elif isinstance(t, TypeScheme):
        new_op = apply(s, t.type_op)
        new_set = apply(s, set(t.env.values()))
        # # todo: apply to typescheme
        # q_types = []
        # for tp in t.quantified_types:
        #     q_types.append(apply(s, tp))
        # tp_scheme = TypeScheme(t.name, q_types, apply(s, t.type_op))
        tp_scheme = generalize(gen_env_dict(new_set), new_op)
        log_content("Apply t: {}".format(str(t)))
        log_content("Apply s: {}".format(str(s)))
        log_content("Apply tp_scheme: {}".format(str(tp_scheme)))
        return tp_scheme
    elif isinstance(t, TypeVariable):
        return s.get(t.name, t)
    elif isinstance(t, set):
        new_set = set()
        for item in t:
            new_set.add(apply(s, item))
        return new_set
    else:
        return t


def applyList(s, xs):
    """ Apply substitution on a list of constraints
    :param s: substitution
    :param xs: list of constraints
    :return: new list
    """
    res_list = []
    for cons in xs:
        if cons.ctype <= ConsType.ConsLess:
            res_list.append(TypeConstraint(apply(s, cons.lhs), apply(s, cons.rhs), cons.ctype))
        else:
            res_list.append(TypeConstraint(apply(s, cons.lhs), apply(s, cons.rhs), cons.ctype, apply(s, cons.mid)))
    # for res in xs:
    #     log_content("pre res:{}".format(res))
    # for res in res_list:
    #     log_content("res:{}".format(res))
    return res_list if isinstance(xs, list) else set(res_list)


def unify(x, y):
    if isinstance(x, Apply) and isinstance(y, Apply):
        s1 = unify(x.fn, y.fn)
        s2 = unify(apply(s1, x.arg), apply(s1, y.arg))
        return compose(s2, s1)
    elif const_type(x) and const_type(y):
        if x.name == y.name:
            return empty()
        else:
            raise InferenceError("Type mismatch: {} and {}".format(x, y))
    elif isinstance(x, TypeOperator) and isinstance(y, TypeOperator):
        if len(x.types) != len(y.types):
            raise InferenceError("Wrong number of arguments")
        s1 = unify(x.types[0], y.types[0])
        s2 = unify(apply(s1, x.types[1]), apply(s1, y.types[1]))
        # log_content("s2: {}".format(str(s2)))
        return compose(s2, s1)
    elif isinstance(x, TypeVariable):
        return bind(x.name, y)
    elif isinstance(y, TypeVariable):
        return bind(y.name, x)
    else:
        raise InferenceError('\n'.join([
            "Type mismatch: ",
            "Given: ", "\t" + str(x),
            "Expected: ", "\t" + str(y)
        ]))


def split_cons(cons):
    if len(cons) <= 1:
        return list(cons)[0], set()
    else:
        next_le_cons = None
        next_m_cons = None
        def get_remain_cons(cur_con, cur_cons):
            res = []
            for tmp in cur_cons:
                if tmp != cur_con:
                    res.append(tmp)
            return res
        for con in cons:
            if con.ctype == ConsType.ConsEq:
                cons.remove(con)
                return con, cons
            elif con.ctype == ConsType.ConsLess:
                next_le_cons = con if next_le_cons is None else next_le_cons
            else:
                next_m_cons = con if next_m_cons is None and con.satisfied(get_remain_cons(con, cons)) else next_m_cons
        next_con = next_m_cons if next_m_cons is not None else next_le_cons
        cons.remove(next_con)
        return next_con, cons


def generalize(env, x):
    if isinstance(x, TypeOperator):
        return x.generalize(env)
    else:
        return x


def gen_env_dict(m_set):
    res = {}
    for m in m_set:
        res[m.name] = m
    return res


def solve_cons(cons):
    # log_content("solve_cons cons:{}".format(cons))
    # for con in cons:
    #     log_content(con)
    if len(cons) == 0:
        s = dict()
    else:
        next_con, remain_cons = split_cons(cons)
        # log_content("next_con: {}".format(next_con))
        if next_con.ctype == ConsType.ConsEq:
            mgu = unify(next_con.lhs, next_con.rhs)
            # log_content("size:{}, cur mgu: {{".format(len(remain_cons)))
            # log_dict(mgu)
            # log_content("}")
            s = compose(solve_cons(applyList(mgu, remain_cons)), mgu)
        elif next_con.ctype == ConsType.ConsLessM:
            if next_con.satisfied(remain_cons):
                type_scheme = generalize(gen_env_dict(next_con.mid), next_con.rhs)
                if type_scheme.empty_quantified():
                    new_cons = TypeConstraint(next_con.lhs, type_scheme.type_op, ConsType.ConsEq)
                else:
                    new_cons = TypeConstraint(next_con.lhs, type_scheme, ConsType.ConsLess)
                log_content("New cons: {}".format(str(new_cons)))
                remain_cons.add(new_cons)
                log_cons(remain_cons)  # log
                s = solve_cons(remain_cons)
            else:
                assert False, "Need to check"
        else:
            new_cons = TypeConstraint(next_con.lhs, instantiate(next_con.rhs), ConsType.ConsEq)
            log_content("New cons: {}".format(str(new_cons)))
            remain_cons.add(new_cons)
            log_cons(remain_cons)  # log
            s = solve_cons(remain_cons)
    # log_content("current substitution: {}".format(s))
    return s


def difference(lhs, rhs):
    lhs_dict = gen_env_dict(lhs)
    rhs_dict = gen_env_dict(rhs)
    res = set()
    for k in set(lhs_dict) - set(rhs_dict):
        res.add(lhs_dict[k])
    return res


def intersect(lhs, rhs):
    lhs_dict = gen_env_dict(lhs)
    rhs_dict = gen_env_dict(rhs)
    res = set()
    for k in set(lhs_dict).intersection(set(rhs_dict)):
        res.add(lhs_dict[k])
    return res


def union_set(lhs, rhs):
    lhs_dict = gen_env_dict(lhs)
    rhs_dict = gen_env_dict(rhs)
    res = set()
    for k in set(lhs_dict).union(set(rhs_dict)):
        if k in set(lhs_dict):
            res.add(lhs_dict[k])
        else:
            res.add(rhs_dict[k])
    return res


def bind(n, x):
    if x == n:
        return empty()
    elif occurs_check(n, x):
        raise InferenceError("recursive unification: {} and {}".format(n, x))
    else:
        return dict([(n, x)])


def occurs_check(n, x):
    # ftvs =
    # log_content("occurs_check: {}, {}, {}".format(str(n), x, ftvs))
    # if len(ftvs) > 0:
    #     for ftv in ftvs:
    #         log_content("ftv:{}, n:{}".format(ftv.name, n))
    #         if ftv.name == n:
    #             return True
    # return False
    return any(n == t.name for t in free_type_variable(x))


def union(s1, s2):
    nenv = s1.copy()
    nenv.update(s2)
    return nenv


def compose(s1, s2):
    s3 = dict((t, apply(s1, u)) for t, u in s2.items())
    # log_content("s1:{}, s2:{}, s3:{}".format(s1, s2, union(s1, s3)))
    return union(s1, s3)


def is_integer_literal(name):
    result = True
    try:
        int(name)
    except ValueError:
        result = False
    return result


def log_dict(dict):
    for e in dict:
        log_content("{}: {}".format(e, dict[e]))


def log_cons(cons):
    log_content("Current cons: {}".format(len(cons)))
    for con in cons:
        log_content(con)
# ==================================================================#
# Example code to exercise the above
def infer_exp(env, node):
    log_perm("node info: {}".format(str(node)))
    log_content("Top env:")
    log_dict(env)
    log_content("")
    global constraint
    constraint = []
    try:
        t, assum, cons = analyse(node, env)
        log_content("\nFinal assumption: {}".format(assum))
        log_content("\nInitial cons: {}".format(len(cons)))
        for item in assum:
            if item not in env:
                raise InferenceError("Undefined variables exist: {}".format(item))
            cons.add(TypeConstraint(assum[item], env[item], ConsType.ConsEq))
        log_content("\nCurrent cons: {}".format(len(cons)))
        for con in cons:
            log_content(con)
        mgu = solve_cons(cons)
        log_content("mgu: {")
        log_dict(mgu)
        log_content("}")
        infer_ty = apply(mgu, t)
        log_content("Inferred type str: {}".format(str(t)))
        log_perm("Inferred value: {}".format(infer_ty))
        return infer_ty, mgu
    except (ParseError, InferenceError) as e:
        log_content(e)


def main():
    """The main example program.

    Sets up some predefined types using the type constructors TypeVariable,
    TypeOperator and Function.  Creates a list of example expressions to be
    evaluated. Evaluates the expressions, log_contenting the type or errors arising
    from each.

    Returns:
        None
    """

    var1 = TypeVariable()
    var2 = TypeVariable()
    pair_type = TypeOperator("*", (var1, var2))

    var3 = TypeVariable()
    var4 = TypeVariable()

    my_env = {"pair": generalize({}, Function(var1, Function(var2, pair_type))),
              "true": Bool,
              # "f": Function(TypeVariable(), TypeVariable()),
              "cond": Function(Bool, Function(var3, Function(var3, var3))),
              "zero": Function(Integer, Bool),
              "pred": Function(Integer, Integer),
              "add": Function(Integer, Function(Integer, Integer)),
              "add_double": Function(Double, Function(Double, Double)),
              "test_f": generalize({}, Function(var4, var4)),
              "merge": Function(Integer, Function(Bool, Bool)),
              "aa": generalize({}, Function(var2, Function(TypeVariable(),  Function(TypeVariable(), TypeVariable())))),
              "times": Function(Integer, Function(Integer, Integer))}

    pair = Apply(Apply(Identifier("pair"),
                       Apply(Identifier("f"),
                             Identifier("4"))),
                 Apply(Identifier("f"),
                       Identifier("true")))

    examples = [
        # factorial
        # Letrec("factorial",  # letrec factorial =
        #        Lambda("n",  # fn n =>
        #               Apply(
        #                   Apply(  # cond (zero n) 1
        #                       Apply(Identifier("cond"),  # cond (zero n)
        #                             Apply(Identifier("zero"), Identifier("n"))),
        #                       Identifier("1")),
        #                   Apply(  # times n
        #                       Apply(Identifier("times"), Identifier("n")),
        #                       Apply(Identifier("factorial"),
        #                             Apply(Identifier("pred"), Identifier("n")))
        #                   )
        #               )
        #               ),  # in
        #        Apply(Identifier("factorial"), Identifier("5"))
        #        ),

        # Should fail:
        # fn x => (pair(x(3) (x(true)))
        # Lambda("x",
        #        Apply(
        #            Apply(Identifier("pair"),
        #                  Apply(Identifier("x"), Identifier("3"))),
        #            Apply(Identifier("x"), Identifier("3")))),
        # Apply(
        #     Apply(Identifier("aa"), Identifier("3")),
        #     Identifier("3")),

        # pair(f(3), f(true))
        # Apply(
        #     Apply(Identifier("pair"), Apply(Identifier("f"), Identifier("4"))),
        #     Apply(Identifier("f"), Identifier("true"))),

        # let f = (fn x => x) in ((pair (f 4)) (f true))
        # Let("f", Lambda("x", Identifier("x")), pair),

        # Apply(Apply(Identifier("pair"), Identifier("3")), Identifier("true")),
        # Let("f", Lambda("x", Identifier("x")),
        #     Apply(Apply(Identifier("merge"), Apply(Identifier("f"), Identifier("3"))), Apply(Identifier("f"), Identifier("true")))),

        # Apply(Identifier("merge"), Apply(Identifier("test_f"), Identifier("3"))),
        # Apply(Apply(Identifier("merge"), Apply(Identifier("test_f"), Identifier("3"))), Apply(Identifier("test_f"), Identifier("true"))),
        # Apply(Identifier("test_f"), Apply(Identifier("test_f"), Identifier("4"))),

        # fn f => f f (fail)
        # Lambda("f", Apply(Identifier("f"), Identifier("f"))),

        # let g = fn f => 5 in g g
        # Let("g",
        #     Lambda("f", Identifier("5")),
        #     Apply(Identifier("g"), Identifier("g"))),

        # example that demonstrates generic and non-generic variables:
        # fn g => let f = fn x => g in pair (f 3, f true)
        # Lambda("g",
        #        Let("f",
        #            Lambda("x", Identifier("g")),
        #            Apply(
        #                Apply(Identifier("pair"),
        #                      Apply(Identifier("f"), Identifier("3"))
        #                      ),
        #                Apply(Identifier("f"), Identifier("true"))))),

        Let("h",
            Lambda("g",
                   Let("f",
                       Lambda("x", Identifier("g")),
                       Apply(
                           Apply(Identifier("pair"),
                                 Apply(Identifier("f"), Identifier("3"))
                                 ),
                           Apply(Identifier("f"), Identifier("true"))))),
            Apply(Identifier("h"), Identifier("3"))),

        # Function composition
        # fn f (fn g (fn arg (f g arg)))
        # Lambda("f", Lambda("g", Lambda("arg", Apply(Identifier("g"), Apply(Identifier("f"), Identifier("arg"))))))
    ]

    # infer_exp(my_env, Apply(Identifier("zero"), Identifier("4")))
    # infer_exp(my_env, Let("f", Lambda("x", Identifier("x")), Apply(Identifier("f"), Identifier("4"))))
    # infer_exp(my_env, Let("y", Identifier("m"), Let("x", Apply(Identifier("y"), Identifier("4")), Identifier("x"))))
    # infer_exp(my_env, Lambda("m", Let("y", Identifier("m"), Let("x", Apply(Identifier("y"), Identifier("4")), Identifier("x")))))

    # log_content("Top env:")
    # for e in my_env:
    #     log_content("{}: {}".format(e, my_env[e]))
    # log_content("")



    # my_env["x3"] = TypeVariable()
    # infer_exp(my_env, Lambda("x", Lambda("y", Apply(Apply(Identifier("add"), Identifier("x3")), Identifier("3")))))
    for example in examples:
        infer_exp(my_env, example)


if __name__ == '__main__':
    main()
