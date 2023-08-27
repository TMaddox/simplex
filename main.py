import numpy as np
import pandas as pd
import yaml
from prettytable import PrettyTable
from utils import to_float


class LinearOptimizationProblem:
    def __init__(self, objective_type, c, b0, A, relations, b, variable_constraints):
        self.objective_type = objective_type
        self.c = c
        self.b0 = b0
        self.A = A
        self.relations = relations
        self.b = b
        self.variable_constraints = variable_constraints

        self.tableaus = []  # Initialize an empty list to store the tableaus
        self.initialized = False

    @classmethod
    def create_from_yaml(cls, filename):
        with open(filename, "r") as file:
            data = yaml.safe_load(file)

        objective_type = data["objective_type"]
        objective_coeffs = to_float(data["objective"]["coefficients"])
        objective_constant = to_float(data["objective"]["constant"])
        constraints = [
            (
                to_float(constraint["coefficients"]),
                constraint["relation"],
                to_float(constraint["rhs"]),
            )
            for constraint in data["constraints"]
        ]
        variable_constraints = data.get("variable_constraints", {})

        A = np.array([constr[0] for constr in constraints])
        b = np.array([constr[2] for constr in constraints])
        b0 = objective_constant
        c = np.array(objective_coeffs)
        relations = [constr[1] for constr in constraints]

        return cls(objective_type, c, b0, A, relations, b, variable_constraints)

    @staticmethod
    def get_basis_variables(tableau):
        """
        Get the row indices and column names of the basic variables in the current tableau.

        Parameters:
        - tableau_df (pandas DataFrame): The current tableau of the LP problem.

        Returns:
        - basis (list of tuple): Each tuple contains the row index and column name of a basic variable.
        """
        num_rows = tableau.shape[0]
        basis = []

        # Iterate through columns (excluding the RHS column)
        for col_name in tableau.columns[:-1]:
            col = tableau[col_name]

            # Check if the column has exactly one entry of 1 and all others are 0
            if (col == 1).sum() == 1 and (col == 0).sum() == num_rows - 1:
                # Get the row index of the 1 entry in the column
                row_idx = col[col == 1].index[0]
                basis.append((row_idx, col_name))

        return basis

    @staticmethod
    def extract_solution(tableau):
        basis_vars = LinearOptimizationProblem.get_basis_variables(tableau)
        basis_vars_dict = {col: row for row, col in basis_vars}
        solution = {}

        # For each column in the tableau, fetch its value if it's a basic variable
        for col_name in tableau.columns[:-1]:  # excluding the "RHS" column
            row_idx = basis_vars_dict.get(col_name)
            if row_idx is not None:
                solution[col_name] = tableau.at[row_idx, "RHS"]
            else:
                solution[col_name] = 0

        return solution

    @staticmethod
    def get_solution_str(solution, show_non_basic_vars=True, rounding_accuracy=2):
        # Filter out non-basic variables if requested
        if not show_non_basic_vars:
            solution = {k: v for k, v in solution.items() if v != 0}

        # Extract and format the keys and values based on the keys in the filtered solution dictionary
        keys = list(solution.keys())
        values = [round(solution[key], rounding_accuracy) for key in keys]

        # Format values, adding underscores to non-zero values
        formatted_values = [
            f"\033[4m{value}\033[0m" if value != 0 else str(value) for value in values
        ]

        # Combine variable names and their values for display
        return f"({' '.join(keys)}) = ({' '.join(formatted_values)})"

    def create_tableau(
        self, include_helper=False
    ):  # TODO function to figure out if helpers are needed
        n_vars = self.A.shape[1]
        n_constraints = self.A.shape[0]
        n_slack = sum(1 for relation in self.relations if relation != "=")
        n_helper = sum(1 for relation in self.relations if relation == "=" or relation == ">=")

        # Initialize an empty DataFrame
        tableau = pd.DataFrame()

        # Add columns for variables
        for i in range(1, n_vars + 1):
            tableau[f"x{i}"] = self.A[:, i - 1]

        # Add columns for slack variables
        i_slack = 1
        for idx, relation in enumerate(self.relations):
            if relation == "<=":
                tableau[f"s{i_slack}"] = [1 if j == idx else 0 for j in range(n_constraints)]
                i_slack += 1
            elif relation == ">=":
                tableau[f"s{i_slack}"] = [-1 if j == idx else 0 for j in range(n_constraints)]
                i_slack += 1
            # no slack variable for equality constraints

        # Add RHS column
        tableau["RHS"] = self.b

        # Add objective function
        tableau.loc[-1] = [-coeff for coeff in self.c] + [0] * n_slack + [self.b0]
        tableau.index = tableau.index + 1
        tableau.sort_index(inplace=True)
        tableau.insert(0, "x0", [1] + [0] * n_constraints)

        if include_helper:
            # Add columns for helper variables
            i_helper = 1
            for idx, relation in enumerate(self.relations):
                if relation == "=":
                    tableau.insert(
                        tableau.shape[1] - 1,
                        f"h{i_helper}",
                        [0] + [1 if j == idx else 0 for j in range(n_constraints)],
                    )
                    i_helper += 1
                elif relation == ">=":
                    tableau.insert(
                        tableau.shape[1] - 1,
                        f"h{i_helper}",
                        [0] + [1 if j == idx else 0 for j in range(n_constraints)],
                    )
                    i_helper += 1
                # no helper variable for <= constraints

            tableau.insert(0, "x-1", [0] + [0] * n_constraints)

            # Add row for helper objective function
            helper_obj_row = pd.Series(
                [0] * tableau.shape[1], dtype="float64", index=tableau.columns
            )
            for hdx in range(1, n_helper + 1):
                for row in range(1, n_constraints + 1):
                    if tableau[f"h{hdx}"][row] == 1:
                        helper_obj_row -= tableau.iloc[row]

            helper_obj_row["x-1"] = 1
            for hdx in range(1, n_helper + 1):
                helper_obj_row[f"h{hdx}"] = 0
            tableau.loc[-1] = helper_obj_row
            tableau.index = tableau.index + 1
            tableau.sort_index(inplace=True)

        self.tableaus.append(tableau)  # Append the tableau to the list of tableaus

    def check_for_optimality(self):
        """
        Check if the current tableau represents an optimal solution.

        Parameters:
        - tableau (pandas DataFrame): The current tableau of the LP problem.

        Returns:
        - bool: True if the solution is optimal, False otherwise.
        """
        # The solution is optimal if all coefficients in the first row (objective function row) are non-negative
        return self.tableaus[-1].iloc[0, :-1].ge(0).all()

    def pivot(self):
        old_tableau = self.tableaus[-1]

        # choose entering variable
        obj_row = old_tableau.iloc[0, :-1]  # Exclude the RHS value
        if obj_row.min() >= 0:
            return None  # If all coefficients are non-negative, the solution is optimal
        idx_entering = obj_row.idxmin()  # Return the column name of the most negative coefficient

        # choose departing variable
        positive_rows = old_tableau[idx_entering] > 0
        positive_ratios = (
            old_tableau.loc[positive_rows, "RHS"] / old_tableau.loc[positive_rows, idx_entering]
        )
        idx_departing = positive_ratios.idxmin()

        # Create a copy of the tableau to avoid modifying the original tableau
        new_tableau = self.tableaus[-1].copy()

        # Scale the departing row
        pivot_value = new_tableau.loc[idx_departing, idx_entering]
        new_tableau.loc[idx_departing] /= pivot_value

        # Update the other rows
        for i in new_tableau.index:
            # Skip the departing row
            if i != idx_departing:
                factor = new_tableau.loc[i, idx_entering]
                new_tableau.loc[i] -= factor * new_tableau.loc[idx_departing]

        self.tableaus.append(new_tableau)  # Append the tableau to the list of tableaus

    def get_dual(self):
        A_dual = self.A.T
        b_dual = self.c
        c_dual = self.b

        objective_type_dual = "min" if self.objective_type == "max" else "max"

        if self.objective_type == "max":
            variable_constraints_dual = {}
            for idx, relation in enumerate(self.relations):
                if relation == "<=":
                    variable_constraints_dual[f"x{idx+1}"] = ">=0"
                elif relation == ">=":
                    variable_constraints_dual[f"x{idx+1}"] = "<=0"
                elif relation == "=":
                    variable_constraints_dual[f"x{idx+1}"] = "R"

            relations_dual = []
            for constraint in self.variable_constraints.values():
                if constraint == ">=0":
                    relations_dual.append(">=")
                elif constraint == "<=0":
                    relations_dual.append("<=")
                elif constraint == "R":
                    relations_dual.append("=")
        else:
            variable_constraints_dual = {}
            for idx, relation in enumerate(self.relations):
                if relation == "<=":
                    variable_constraints_dual[f"x{idx+1}"] = "<=0"
                elif relation == ">=":
                    variable_constraints_dual[f"x{idx+1}"] = ">=0"
                elif relation == "=":
                    variable_constraints_dual[f"x{idx+1}"] = "R"

            relations_dual = []
            for constraint in self.variable_constraints.values():
                if constraint == ">=0":
                    relations_dual.append("<=")
                elif constraint == "<=0":
                    relations_dual.append(">=")
                elif constraint == "R":
                    relations_dual.append("=")

        return LinearOptimizationProblem(
            objective_type_dual,
            c_dual,
            self.b0,
            A_dual,
            relations_dual,
            b_dual,
            variable_constraints_dual,
        )

    def initialize(self):
        if not self.initialized:
            self.create_tableau()  # Create the initial tableau
            self.initialized = True

    def solve(self):
        optimality = self.check_for_optimality()
        while not optimality:
            self.pivot()
            optimality = self.check_for_optimality()

    def get_constraints_str(self):
        """
        Generate a string representation of the constraints.

        Returns:
        - str: String representation of the constraints using PrettyTable.
        """
        num_coeffs = len(self.c)

        # Create a new pretty table
        pt = PrettyTable()

        # Set the column names
        header = ["Coeff" + str(i + 1) for i in range(num_coeffs)] + ["Relation", "RHS"]
        pt.field_names = header

        # Adding the objective coefficients row
        obj_row = ["{:.2f}".format(val) for val in self.c] + [
            "Obj. Func.",
            "{:.2f}".format(self.b0),
        ]
        pt.add_row(obj_row)

        # Rows with dynamic width for constraints
        for i in range(self.A.shape[0]):
            coeffs = self.A[i]
            relation = self.relations[i]
            rhs = self.b[i]
            formatted_row = ["{:.2f}".format(val) for val in coeffs] + [
                relation,
                "{:.2f}".format(rhs),
            ]
            pt.add_row(formatted_row)

            # Row for variable constraints
        variable_constraints_row = []
        for i in range(1, num_coeffs + 1):
            constraint = self.variable_constraints.get(f"x{i}", "-")
            variable_constraints_row.append(constraint)
        variable_constraints_row += ["", ""]
        pt.add_row(variable_constraints_row)

        return str(pt)

    def get_tableau_str_by_idx(
        self, idx, show_basic_vars=True, rounding_accuracy=2, display_basicsolution=False
    ):
        # Getting the basis variables (row, col) pairs
        basis_vars = self.get_basis_variables(self.tableaus[idx])
        basis_rows = {row: col for row, col in basis_vars}

        rows, cols = self.tableaus[idx].shape

        # Filter columns based on whether basic variables should be displayed
        if show_basic_vars:
            valid_cols = set(self.tableaus[idx].columns)
        else:
            valid_cols = set(self.tableaus[idx].columns) - set(col for _, col in basis_vars)

        # Create a new pretty table
        pt = PrettyTable()

        # Set the column names
        header = ["Basis"] + [col for col in self.tableaus[idx].columns if col in valid_cols]
        pt.field_names = header

        # Add rows to the table
        for i in range(rows):
            basis_col = basis_rows.get(i, "-")  # Using "-" for non-existent basis
            formatted_row = [basis_col] + [
                round(self.tableaus[idx].iloc[i][col], rounding_accuracy)
                for col in self.tableaus[idx].columns
                if col in valid_cols
            ]
            pt.add_row(formatted_row)

        tableau_str = str(pt)

        if display_basicsolution:
            solution_str = self.get_solution_str(
                LinearOptimizationProblem.extract_solution(self.tableaus[idx])
            )
            tableau_str += f"\nBasic solution: {solution_str}"

        return tableau_str

    def get_tableau_str_latest(
        self, show_basic_vars=True, rounding_accuracy=2, display_basicsolution=False
    ):
        return self.get_tableau_str_by_idx(
            -1, show_basic_vars, rounding_accuracy, display_basicsolution
        )

    def get_tableau_str_all(
        self, show_basic_vars=True, rounding_accuracy=2, display_basicsolution=False
    ):
        tableau_str = ""
        for idx in range(len(self.tableaus)):
            tableau_str += f"\nTableau {idx}:\n"
            tableau_str += self.get_tableau_str_by_idx(
                idx, show_basic_vars, rounding_accuracy, display_basicsolution
            )
            tableau_str += "\n"
        return tableau_str


if __name__ == "__main__":
    print(chr(27) + "[2J")
    print("\n*** START ***")
    in_file = "in.yaml"
    print(f"Read from file: {in_file}")
    LOP = LinearOptimizationProblem.create_from_yaml(in_file)

    print("Constraints:")
    print(LOP.get_constraints_str())

    # simple solve
    # LOP.solve()
    # print(LOP.get_tableau_str_all(display_basicsolution=True))

    LOP_dual = LOP.get_dual()
    print("\nDual problem:")
    print(LOP_dual.get_constraints_str())

    print("\n*** END ***")
