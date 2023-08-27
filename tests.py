import numpy as np
import unittest
import os

from main import *


class TestImportFromFile(unittest.TestCase):
    def setUp(self):
        # Create a sample input file for testing
        self.filename = "test_input.txt"
        with open(self.filename, "w") as f:
            f.write("max\n")
            f.write("5/2 0 -3+1/3 -4.5\n")  # Using fractions and expressions
            f.write("50 0 20 <= 200\n")
            f.write("4 2 0 >= 30\n")  # Greater than or equal to inequality
            f.write("-60 0 -20 = 480\n")  # Negative coefficients and equality
            f.write("0 0 0 <= 50\n")  # Zero coefficients

    def tearDown(self):
        # Cleanup: Remove the sample input file after the test
        os.remove(self.filename)

    def test_import_from_file(self):
        objective_type, objective_coeffs, objective_constant, constraints = import_from_file(
            self.filename
        )

        # Validate the extracted data
        self.assertEqual(objective_type, "max")
        self.assertEqual(objective_coeffs, [2.5, 0, -3 + 1 / 3])
        self.assertEqual(objective_constant, -4.5)
        self.assertEqual(
            constraints,
            [
                ([50, 0, 20], "<=", 200),
                ([4, 2, 0], ">=", 30),
                ([-60, 0, -20], "=", 480),
                ([0, 0, 0], "<=", 50),
            ],
        )


class TestExtractMatrices(unittest.TestCase):
    def setUp(self):
        self.objective_type = "min"
        self.objective_coeffs = [1 / 2, -3, 0, 7, -2.5]
        self.objective_constant = 1.5
        self.constraints = [
            ([2 / 3, 0, -1, 5, -3], "<=", 15.5),
            ([0, -3 / 4, 0, 0, 5], ">=", -2),
            ([1, 1, 1, 1, 1], "=", 5),
            ([-1, 2, -1, 2, -1], "<=", 0),
            ([-3, 0, 2, 7, 0], ">=", 8.5),
        ]

    def test_extract_matrices(self):
        A, b, b0, c, relations, obj_type = extract_matrices(
            self.objective_type, self.objective_coeffs, self.objective_constant, self.constraints
        )

        # Validate the extracted matrices and vectors
        expected_A = np.array(
            [
                [2 / 3, 0, -1, 5, -3],
                [0, -3 / 4, 0, 0, 5],
                [1, 1, 1, 1, 1],
                [-1, 2, -1, 2, -1],
                [-3, 0, 2, 7, 0],
            ]
        )
        np.testing.assert_array_equal(A, expected_A)

        expected_b = np.array([15.5, -2, 5, 0, 8.5])
        np.testing.assert_array_equal(b, expected_b)

        self.assertEqual(b0, 1.5)

        expected_c = np.array([1 / 2, -3, 0, 7, -2.5])
        np.testing.assert_array_equal(c, expected_c)

        self.assertEqual(relations, ["<=", ">=", "=", "<=", ">="])
        self.assertEqual(obj_type, "min")


class TestGetDual(unittest.TestCase):
    def test_complex_scenario(self):
        A = np.array([[2, 3, 1], [1, 0, 0], [0, 2, 3]])
        b = np.array([5, 2, 4])
        b0 = 10
        c = np.array([1, 2, 1])
        relations = ["<=", "=", ">="]
        primal_objective_type = "max"

        A_dual_expected = np.array([[2, 1, 0], [3, 0, 2], [1, 0, 3]])
        b_dual_expected = c
        c_dual_expected = b
        relations_dual_expected = [">=", ">=", "<="]
        dual_objective_type_expected = "min"

        A_dual, b_dual, b0_dual, c_dual, relations_dual, dual_objective_type = get_dual(
            A, b, b0, c, relations, primal_objective_type
        )

        np.testing.assert_array_equal(A_dual, A_dual_expected)
        np.testing.assert_array_equal(b_dual, b_dual_expected)
        np.testing.assert_array_equal(c_dual, c_dual_expected)
        self.assertEqual(relations_dual, relations_dual_expected)
        self.assertEqual(dual_objective_type, dual_objective_type_expected)
        self.assertEqual(b0_dual, -10)

    def test_exercise5_1_i(self):
        A = np.array([[2, -3, 1], [5, 0, 6]])
        b = np.array([2, -1])
        b0 = 0
        c = np.array([5, 2, -1])
        relations = ["<=", "<="]
        primal_objective_type = "max"

        A_dual_expected = np.array([[2, 5], [-3, 0], [1, 6]])
        b_dual_expected = np.array([5, 2, -1])
        c_dual_expected = np.array([2, -1])
        relations_dual_expected = [">=", ">="]
        dual_objective_type_expected = "min"

        A_dual, b_dual, b0_dual, c_dual, relations_dual, dual_objective_type = get_dual(
            A, b, b0, c, relations, primal_objective_type
        )

        np.testing.assert_array_equal(A_dual, A_dual_expected)
        np.testing.assert_array_equal(b_dual, b_dual_expected)
        np.testing.assert_array_equal(c_dual, c_dual_expected)
        self.assertEqual(relations_dual, relations_dual_expected)
        self.assertEqual(dual_objective_type, dual_objective_type_expected)

    def test_exercise5_1_ii(self):
        A = np.array([[3, -1, 1], [4, -2, 6]])
        b = np.array([2, 3])
        b0 = 0
        c = np.array([2, -1, 5])
        relations = ["=", "="]
        primal_objective_type = "min"

        A_dual_expected = np.array([[3, 4], [-1, -2], [1, 6]])
        b_dual_expected = np.array([2, -1, 5])
        c_dual_expected = np.array([2, 3])
        relations_dual_expected = ["<=", "<="]
        dual_objective_type_expected = "max"

        A_dual, b_dual, b0_dual, c_dual, relations_dual, dual_objective_type = get_dual(
            A, b, b0, c, relations, primal_objective_type
        )

        np.testing.assert_array_equal(A_dual, A_dual_expected)
        np.testing.assert_array_equal(b_dual, b_dual_expected)
        np.testing.assert_array_equal(c_dual, c_dual_expected)
        self.assertEqual(relations_dual, relations_dual_expected)
        self.assertEqual(dual_objective_type, dual_objective_type_expected)


if __name__ == "__main__":
    unittest.main()
