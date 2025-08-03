import argparse


def main(nums=[], k=0):
    def gen_inner_array():
        for i in range(0, len(nums) - k + 1):
            yield max(nums[i : i + k])

    res = [x for x in gen_inner_array()]
    print(res)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int)
    parser.add_argument("-n", "--nums", type=int, nargs="+")
    args = parser.parse_args().__dict__
    main(**args)
