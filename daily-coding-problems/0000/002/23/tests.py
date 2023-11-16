import unittest

from .solution import compute_all_paths, main, valid_moves, \
    generate_moves, tile_is_within_matrix_limits, tile_is_not_wall


class TestCase(unittest.TestCase):
    def setUp(self):
        self.basic_matrix = [[False, False], [False, False]]
        self.given_matrix = [[False, False, False, False],
                             [True, True, False, True],
                             [False, False, False, False],
                             [False, False, False, False]]

    def test_tile_is_not_wall(self):
        self.assertTrue(tile_is_not_wall(
            tile=(0, 0), matrix=self.basic_matrix))
        self.assertTrue(tile_is_not_wall(
            tile=(1, 2), matrix=self.given_matrix))

    def test_tile_is_wall(self):
        self.assertFalse(tile_is_not_wall(
            tile=(1, 0), matrix=self.given_matrix))
        self.assertFalse(tile_is_not_wall(
            tile=(1, 3), matrix=self.given_matrix))

    def test_tile_is_within_matrix_limits(self):
        self.assertTrue(tile_is_within_matrix_limits(
            tile=(0, 0), matrix=self.basic_matrix))
        self.assertTrue(tile_is_within_matrix_limits(
            tile=(0, 1), matrix=self.basic_matrix))
        self.assertTrue(tile_is_within_matrix_limits(
            tile=(1, 0), matrix=self.basic_matrix))
        self.assertTrue(tile_is_within_matrix_limits(
            tile=(1, 1), matrix=self.basic_matrix))

    def test_tile_is_not_within_matrix_limits(self):
        self.assertFalse(tile_is_within_matrix_limits(
            tile=(-1, 0), matrix=self.basic_matrix))
        self.assertFalse(tile_is_within_matrix_limits(
            tile=(0, 10), matrix=self.basic_matrix))
        self.assertFalse(tile_is_within_matrix_limits(
            tile=(100, 100), matrix=self.basic_matrix))
        self.assertFalse(tile_is_within_matrix_limits(
            tile=(-1, -10), matrix=self.basic_matrix))

    def test_given(self):
        given_input = {'start': (3, 0),
                       'end': (0, 0),
                       'matrix': self.given_matrix}
        expected_result = 7
        computed_result = main(**given_input)
        self.assertEqual(expected_result, computed_result)

    def test_valid_moves_with_basic_matrix(self):
        expected_result = [(1, 0), (0, 1)]
        computed_result = valid_moves((0, 0), self.basic_matrix)
        self.assertEqual(expected_result, computed_result)

    def test_valid_moves_with_given_matrix(self):
        expected_result = [(0, 2), (2, 2)]
        computed_result = valid_moves((1, 2), self.given_matrix)
        self.assertEqual(expected_result, computed_result)

    def test_compute_all_paths(self):
        expected_result = [[(0, 0), (0, 1), (1, 1)], [(0, 0), (1, 0), (1, 1)]]
        computed_result = compute_all_paths(start=(0, 0),
                                            end=(1, 1),
                                            matrix=self.basic_matrix)
        assert computed_result is not None
        self.assertEqual(sorted(expected_result), sorted(computed_result))

    def test_generate_moves(self):
        expected_result = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        computed_result = list(generate_moves((0, 0)))
        self.assertEqual(expected_result, computed_result)


if __name__ == "__main__":
    unittest.main()
