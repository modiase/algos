from loguru import logger


def get_msb(n: int) -> int:
    n = n if n > 0 else -n
    # First propagate rightmost 1-bit to the right
    shift = 1
    while n >> shift:  # Continue while there are still 1 bits to propagate
        logger.debug(f"{n:b}")
        logger.debug(f"{n >> shift:b}\n")
        n |= n >> shift
        shift <<= 1  # Double the shift amount each time

    # Then get the MSB by subtracting the right shift by 1
    if n:
        return n - (n >> 1)
    return 0


if __name__ == "__main__":
    logger.info(get_msb(10))
    logger.info(get_msb(1023))
    logger.info(get_msb(1024))
    logger.info(get_msb(-1024))
