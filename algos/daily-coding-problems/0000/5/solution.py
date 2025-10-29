import typing


def car(cons_func: typing.Callable):
    def _exec(a, b):
        return (a, b)

    return cons_func(_exec)[0]


def cdr(cons_func: typing.Callable):
    def _exec(a, b):
        return (a, b)

    return cons_func(_exec)[-1]
