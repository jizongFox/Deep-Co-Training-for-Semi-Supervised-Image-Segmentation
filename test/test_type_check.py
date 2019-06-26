import inspect
from typing import Union

import torch
from typing_inspect import is_union_type

from generalframework.utils import one_hot, predlogit2one_hot


def accepts(func):
    types = func.__annotations__
    for k, v in types.items():
        if is_union_type(v):
            types[k] = tuple(v.__args__)

    def check_accepts(*args, **kwargs):
        for (a, t) in zip(args, list(types.values())):
            assert isinstance(a, t), \
                "arg %r does not match %s" % (a, t)

        for k, v in kwargs.items():
            assert isinstance(v, types[k]), \
                f'kwargs {k}:{v} does not match {types[k]}'

        return func(*args, **kwargs)

    return check_accepts


def onehot(name):
    assert isinstance(name, (str, list))

    def check_onehot(f):
        f_sig = inspect.signature(f)
        if isinstance(name, str):
            assert name in f_sig.parameters.keys()
        else:
            assert set(f_sig.parameters.keys()).issuperset(
                set(name)), f'{name} should be included in {list(f_sig.parameters.keys())}'

        def new_f(*args, **kwds):
            for (a, t) in zip(args, f_sig.parameters.keys()):
                if t == name or t in name:
                    assert one_hot(a, 1), f'{t}={a} onehot check failed'

            for k, v in kwds.items():
                if k == name or k in name:
                    assert one_hot(v, 1), f'{k}={v} onehot check failed'
            res = f(*args,**kwds)

            return res

        new_f.__name__ = f.__name__
        return new_f

    return check_onehot




def test_onehot_check():
    @onehot(['b','c'])
    def count_function(a, b, c):
        pass

    # count_function(torch.Tensor([1]), predlogit2one_hot(torch.randn(4, 2, 256, 256)), True)
    count_function(torch.Tensor([1]), b=predlogit2one_hot(torch.randn(4, 2, 256, 256)), c=predlogit2one_hot(torch.randn(4, 2, 256, 256)))


def test_accept_check():
    @accepts
    def count_function(a: int, b: str, c: Union[list, str]):
        pass

    count_function(100, c=[1, 2, 3], b='okay')


if __name__ == '__main__':
    test_accept_check()
    test_onehot_check()
