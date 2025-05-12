"""
What will this code print out?

def make_functions():
    flist = []

    for i in [1, 2, 3]:
        def print_i():
            print(i)
        flist.append(print_i)

    return flist

functions = make_functions()
for f in functions:
    f()
"""

# The above prints 3, 3, 3


def make_functions():
    flist = []

    for i in [1, 2, 3]:
        def print_i(x):
            return lambda: print(x)
        flist.append(print_i(i))

    return flist


functions = make_functions()
for f in functions:
    f()
