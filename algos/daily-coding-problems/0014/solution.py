import random as rn


def throw():
    x = rn.random()
    y = rn.random()
    return pow(x - 0.5, 2) + pow(y - 0.5, 2) < 0.25


def main():
    N = 10000000
    counter = 0
    for i in range(0, N):
        if throw():
            counter += 1
    return 4 * counter / float(N)


if __name__ == "__main__":
    PI = main()
    print(PI)
