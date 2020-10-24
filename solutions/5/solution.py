from given import cons

import typing


def car(cons: typing.Callable):
    def _exec(a, b):
        return (a, b)
    return cons(_exec)[0]


def cdr(cons: typing.Callable):
    def _exec(a, b):
        return (a, b)
    return cons(_exec)[-1]
