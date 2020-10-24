from solution import main


def test_solution():
    results = []
    for i in range(0, 1000):
        res = main(10)
        results.append(res)
    assert int(round(sum(results) / len(results))) == 5
