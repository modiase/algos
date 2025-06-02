import ctypes

from loguru import logger


def twos_complement(n):
    return ctypes.c_int32(~n + 1).value


def subtract_by_addition(a, b):
    return ctypes.c_int32(a + twos_complement(b)).value


if __name__ == "__main__":
    logger.info(subtract_by_addition(10, 5))
    logger.info(subtract_by_addition(5, 10))
