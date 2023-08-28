import os
import numpy as np
import yaml
import unittest

from main import LinearOptimizationProblem


class TestImportFromYAML(unittest.TestCase):
    def setUp(self):
        # Create a sample input file for testing
        self.filename = "test_input.yml"
        data = {
            "objective_type": "max",
            "objective": {"coefficients": ["5/2", "0", "-1/3"], "constant": "-4.5"},
            "constraints": [
                {"coefficients": ["50", "0", "20"], "relation": "<=", "rhs": "200"},
                {"coefficients": ["4", "2", "0"], "relation": ">=", "rhs": "30"},
                {"coefficients": ["-60", "0", "-20"], "relation": "=", "rhs": "480"},
                {"coefficients": ["0", "0", "0"], "relation": "<=", "rhs": "50"},
            ],
            "variable_constraints": {"x1": "R", "x2": ">=0"},
        }
        with open(self.filename, "w") as f:
            yaml.dump(data, f)

    def tearDown(self):
        # Cleanup: Remove the sample input file after the test
        os.remove(self.filename)

    def test_import_from_yaml(self):
        LOP = LinearOptimizationProblem.create_from_yaml(self.filename)

        # Validate the extracted data
        self.assertEqual(LOP.objective_type, "max")
        self.assertTrue((LOP.c == [2.5, 0, -1 / 3]).all())
        self.assertEqual(LOP.b0, -4.5)
        self.assertTrue(
            (LOP.A == np.array([[50, 0, 20], [4, 2, 0], [-60, 0, -20], [0, 0, 0]])).all()
        )
        self.assertEqual(LOP.relations, ["<=", ">=", "=", "<="])
        self.assertTrue((LOP.b == np.array([200, 30, 480, 50])).all())
        self.assertEqual(LOP.variable_constraints, {"x1": "R", "x2": ">=0"})


class LinearOptimizationProblemTests(unittest.TestCase):
    def assertDictAlmostEqual(self, d1, d2, places=7):
        """Assert that two dictionaries are almost equal."""
        self.assertEqual(d1.keys(), d2.keys())
        for k, v1 in d1.items():
            v2 = d2[k]
            self.assertAlmostEqual(v1, v2, places=places)

    def test_exercise5_1_i(self):
        primal_objective_type = "max"
        c = np.array([5, 2, -1])
        b0 = 0
        A = np.array([[2, -3, 1], [5, 0, 6]])
        relations = ["<=", "<="]
        b = np.array([2, -1])
        variable_constraints = {"x1": ">=0", "x2": ">=0", "x3": ">=0"}

        LOP = LinearOptimizationProblem(
            primal_objective_type, c, b0, A, relations, b, variable_constraints
        )

        dual_objective_type_expected = "min"
        c_dual_expected = np.array([2, -1])
        b0_dual_expected = 0
        A_dual_expected = np.array([[2, 5], [-3, 0], [1, 6]])
        relations_dual_expected = [">=", ">=", ">="]
        b_dual_expected = np.array([5, 2, -1])
        variable_constraints_dual_expected = {"x1": ">=0", "x2": ">=0"}

        LOP_dual = LOP.get_dual()

        self.assertEqual(LOP_dual.objective_type, dual_objective_type_expected)
        np.testing.assert_array_equal(LOP_dual.c, c_dual_expected)
        self.assertEqual(LOP_dual.b0, b0_dual_expected)
        np.testing.assert_array_equal(LOP_dual.A, A_dual_expected)
        self.assertEqual(LOP_dual.relations, relations_dual_expected)
        np.testing.assert_array_equal(LOP_dual.b, b_dual_expected)
        self.assertEqual(LOP_dual.variable_constraints, variable_constraints_dual_expected)

    def test_exercise5_1_ii(self):
        primal_objective_type = "min"
        c = np.array([2, -1, 5])
        b0 = 0
        A = np.array([[3, -1, 1], [4, -2, 6]])
        relations = ["=", "="]
        b = np.array([2, 3])
        variable_constraints = {"x1": ">=0", "x2": ">=0", "x3": ">=0"}

        LOP = LinearOptimizationProblem(
            primal_objective_type, c, b0, A, relations, b, variable_constraints
        )

        dual_objective_type_expected = "max"
        c_dual_expected = np.array([2, 3])
        b0_dual_expected = 0
        A_dual_expected = np.array([[3, 4], [-1, -2], [1, 6]])
        relations_dual_expected = ["<=", "<=", "<="]
        b_dual_expected = np.array([2, -1, 5])
        variable_constraints_dual_expected = {"x1": "R", "x2": "R"}

        LOP_dual = LOP.get_dual()

        self.assertEqual(LOP_dual.objective_type, dual_objective_type_expected)
        np.testing.assert_array_equal(LOP_dual.c, c_dual_expected)
        self.assertEqual(LOP_dual.b0, b0_dual_expected)
        np.testing.assert_array_equal(LOP_dual.A, A_dual_expected)
        self.assertEqual(LOP_dual.relations, relations_dual_expected)
        np.testing.assert_array_equal(LOP_dual.b, b_dual_expected)
        self.assertEqual(LOP_dual.variable_constraints, variable_constraints_dual_expected)

    def test_exercise5_1_iii(self):
        primal_objective_type = "min"
        c = np.array([2, 3])
        b0 = 0
        A = np.array([[1, -1], [2, -1], [-1, 3]])
        relations = [">=", ">=", ">="]
        b = np.array([2, 4, -1])
        variable_constraints = {"x1": "R", "x2": "R"}

        LOP = LinearOptimizationProblem(
            primal_objective_type, c, b0, A, relations, b, variable_constraints
        )

        dual_objective_type_expected = "max"
        c_dual_expected = np.array([2, 4, -1])
        b0_dual_expected = 0
        A_dual_expected = np.array([[1, 2, -1], [-1, -1, 3]])
        relations_dual_expected = ["=", "="]
        b_dual_expected = np.array([2, 3])
        variable_constraints_dual_expected = {"x1": ">=0", "x2": ">=0", "x3": ">=0"}

        LOP_dual = LOP.get_dual()

        self.assertEqual(LOP_dual.objective_type, dual_objective_type_expected)
        np.testing.assert_array_equal(LOP_dual.c, c_dual_expected)
        self.assertEqual(LOP_dual.b0, b0_dual_expected)
        np.testing.assert_array_equal(LOP_dual.A, A_dual_expected)
        self.assertEqual(LOP_dual.relations, relations_dual_expected)
        np.testing.assert_array_equal(LOP_dual.b, b_dual_expected)
        self.assertEqual(LOP_dual.variable_constraints, variable_constraints_dual_expected)

    def test_exercise5_5(self):
        # define primal problem
        primal_objective_type = "max"
        c = np.array([1, 1])
        b0 = 0
        A = np.array([[-1, 2], [3, 2]])
        relations = ["<=", "<="]
        b = np.array([4, 12])
        variable_constraints = {"x1": ">=0", "x2": ">=0"}

        LOP_primal = LinearOptimizationProblem(
            primal_objective_type, c, b0, A, relations, b, variable_constraints
        )

        # get dual problem
        LOP_dual = LOP_primal.get_dual()

        # define expected results dual problem
        dual_objective_type_expected = "min"
        c_dual_expected = np.array([4, 12])
        b0_dual_expected = 0
        A_dual_expected = np.array([[-1, 3], [2, 2]])
        relations_dual_expected = [">=", ">="]
        b_dual_expected = np.array([1, 1])
        variable_constraints_dual_expected = {"x1": ">=0", "x2": ">=0"}

        self.assertEqual(LOP_dual.objective_type, dual_objective_type_expected)
        np.testing.assert_array_equal(LOP_dual.c, c_dual_expected)
        self.assertEqual(LOP_dual.b0, b0_dual_expected)
        np.testing.assert_array_equal(LOP_dual.A, A_dual_expected)
        self.assertEqual(LOP_dual.relations, relations_dual_expected)
        np.testing.assert_array_equal(LOP_dual.b, b_dual_expected)
        self.assertEqual(LOP_dual.variable_constraints, variable_constraints_dual_expected)

        # initialize problems
        LOP_primal.initialize()
        LOP_dual.initialize()

        # solve problems
        LOP_primal.solve()
        LOP_dual.solve()

        # define expected results
        x_primal_expected = {"x0": 5, "x1": 2, "x2": 3}
        x_dual_expected = {"x0": 5, "x1": 1 / 8, "x2": 3 / 8}

        # validate results
        x_primal = {k: v for k, v in LOP_primal.x.items() if k.startswith("x")}
        x_dual = {k: v for k, v in LOP_dual.x.items() if k.startswith("x")}
        self.assertDictAlmostEqual(x_primal, x_primal_expected)
        self.assertDictAlmostEqual(x_dual, x_dual_expected)


if __name__ == "__main__":
    unittest.main()
