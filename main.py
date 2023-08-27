import numpy as np
import pandas as pd
from prettytable import PrettyTable


def import_from_file(filename):
    """
    Imports a linear program from a file and extracts its components.

    The expected format of the file is:
    - First line: Objective type ("max" or "min").
    - Second line: Coefficients of the objective function variables, space-separated,
      with the last coefficient being the constant term of the objective function.
    - Subsequent lines: Coefficients of the constraint variables, space-separated,
      followed by the relation ("<=", ">=", "=") and the right-hand side value of the constraint.

    Parameters:
    - filename (str): Path to the file containing the linear program definition.

    Returns:
    - objective_type (str): The objective type of the LP, either "max" or "min".
    - objective_coeffs (list of float): Coefficients of the objective function variables.
    - objective_constant (float): Constant term from the objective function.
    - constraints (list of tuple): List of constraints, where each constraint is a tuple of
      coefficients, relation, and right-hand side value.

    Raises:
    - AssertionError: If the number of coefficients in any constraint doesn't match
      the number of objective function coefficients, or if an invalid relation is encountered.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

        # Get the objective type
        objective_type = lines[0].strip().lower()

        # Get the coefficients of the objective function
        # Using eval to convert fractions (or any valid expression) into float
        obj_parts = [eval(coeff) for coeff in lines[1].split()]
        objective_coeffs = obj_parts[:-1]
        objective_constant = obj_parts[-1]

        # Get the constraints
        constraints = []
        for line in lines[2:]:
            parts = line.split()

            # Ensure that the number of coefficients in each constraint matches the number of objective coefficients
            assert len(parts[:-2]) == len(
                objective_coeffs
            ), "Mismatch in number of coefficients in constraints and objective function."

            coeffs = [eval(coeff) for coeff in parts[:-2]]
            relation = parts[-2]
            assert relation in ["<=", ">=", "="], f"Invalid relation: {relation} in constraints."
            rhs = float(parts[-1])
            constraints.append((coeffs, relation, rhs))

    return objective_type, objective_coeffs, objective_constant, constraints


def extract_matrices(objective_type, objective_coeffs, objective_constant, constraints):
    """
    Extract the matrices and vectors representing a linear program from its components.

    Parameters:
    - objective_type (str): The objective type of the LP, either "max" or "min".
    - objective_coeffs (list of float): Coefficients of the objective function variables.
    - objective_constant (float): Constant term from the objective function.
    - constraints (list of tuple): List of constraints, where each constraint is a tuple of
    coefficients, relation, and right-hand side value.

    Returns:
    - A (numpy array): The constraint matrix.
    - b (numpy array): The right-hand side vector of the constraints.
    - b0 (float): The constant term from the objective function.
    - c (numpy array): Coefficients of the objective function variables.
    - relations (list of str): List of relations corresponding to each
    constraint (e.g., "<=", ">=", "=").
    - objective_type (str): The objective type of the LP, either "max" or "min".
    """
    A = np.array([constr[0] for constr in constraints])
    b = np.array([constr[2] for constr in constraints])
    b0 = objective_constant
    c = np.array(objective_coeffs)
    relations = [constr[1] for constr in constraints]

    return A, b, b0, c, relations, objective_type


def get_dual(A, b, b0, c, relations, primal_objective_type):
    """
    Compute the dual linear program for a given primal linear program.

    Parameters:
    - A (numpy array): Constraint matrix of the primal problem.
    - b (numpy array): Right-hand side vector of the primal constraints.
    - b0 (float): Constant term from the primal objective function.
    - c (numpy array): Coefficients of the primal objective function variables.
    - relations (list of str): List of relations corresponding to each primal constraint (e.g., "<=", "=").
    - primal_objective_type (str): The objective type of the primal LP, either "max" or "min".

    Returns:
    - A_dual (numpy array): Constraint matrix of the dual problem.
    - b_dual (numpy array): Right-hand side vector of the dual constraints.
    - c_dual (numpy array): Coefficients of the dual objective function variables.
    - relations_dual (list of str): List of relations corresponding to each dual constraint.
    - dual_objective_type (str): The objective type of the dual LP, either "max" or "min".
    """
    A_dual = A.T
    b_dual = c
    c_dual = b

    relations_dual = []
    for relation in relations:
        if relation == "<=":
            relations_dual.append(">=")
        elif relation == ">=":
            relations_dual.append("<=")
        elif relation == "=":
            relations_dual.append(">=")  # Using ">=" to indicate unrestricted "u"

    # If primal is maximization, dual will be minimization and vice-versa.
    dual_objective_type = "min" if primal_objective_type == "max" else "max"

    return A_dual, b_dual, -b0, c_dual, relations_dual, dual_objective_type


def create_tableau(A, b, b0, c, relations, include_helper=False):
    n_vars = A.shape[1]
    n_constraints = A.shape[0]
    n_slack = sum(1 for relation in relations if relation != "=")
    n_helper = sum(1 for relation in relations if relation == "=" or relation == ">=")

    # Initialize an empty DataFrame
    tableau = pd.DataFrame()

    # Add columns for variables
    for i in range(1, n_vars + 1):
        tableau[f"x{i}"] = A[:, i - 1]

    # Add columns for slack variables
    i_slack = 1
    for idx, relation in enumerate(relations):
        if relation == "<=":
            tableau[f"s{i_slack}"] = [1 if j == idx else 0 for j in range(n_constraints)]
            i_slack += 1
        elif relation == ">=":
            tableau[f"s{i_slack}"] = [-1 if j == idx else 0 for j in range(n_constraints)]
            i_slack += 1
        # no slack variable for equality constraints

    # Add RHS column
    tableau["RHS"] = b

    # Add objective function
    tableau.loc[-1] = [-coeff for coeff in c] + [0] * n_slack + [b0]
    tableau.index = tableau.index + 1
    tableau.sort_index(inplace=True)
    tableau.insert(0, "x0", [1] + [0] * n_constraints)

    if include_helper:
        # Add columns for helper variables
        i_helper = 1
        for idx, relation in enumerate(relations):
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
        helper_obj_row = pd.Series([0] * tableau.shape[1], dtype="float64", index=tableau.columns)
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

    return tableau


def setup_initial_tableau(
    objective_type, objective_coeffs, objective_constant, constraints, include_helper=False
):
    # Extract matrices and vectors from the LP problem
    A, b, b0, c, relations, objective_type = extract_matrices(
        objective_type, objective_coeffs, objective_constant, constraints
    )

    # Create the primal tableau
    tableau = create_tableau(A, b, b0, c, relations, include_helper=include_helper)

    # Get the dual problem
    A_d, b_d, b0_d, c_d, relations_d, objective_type_d = get_dual(
        A, b, b0, c, relations, objective_type
    )

    # Create the dual tableau
    tableau_d = create_tableau(A_d, b_d, b0_d, c_d, relations_d, include_helper=include_helper)

    return tableau, tableau_d


def check_for_optimality(tableau):
    """
    Check if the current tableau represents an optimal solution.

    Parameters:
    - tableau (pandas DataFrame): The current tableau of the LP problem.

    Returns:
    - bool: True if the solution is optimal, False otherwise.
    """
    # The solution is optimal if all coefficients in the first row (objective function row) are non-negative
    return tableau.iloc[0, :-1].ge(0).all()


def get_basis_variables(tableau_df):
    """
    Get the row indices and column names of the basic variables in the current tableau.

    Parameters:
    - tableau_df (pandas DataFrame): The current tableau of the LP problem.

    Returns:
    - basis (list of tuple): Each tuple contains the row index and column name of a basic variable.
    """
    num_rows = tableau_df.shape[0]
    basis = []

    # Iterate through columns (excluding the RHS column)
    for col_name in tableau_df.columns[:-1]:
        col = tableau_df[col_name]

        # Check if the column has exactly one entry of 1 and all others are 0
        if (col == 1).sum() == 1 and (col == 0).sum() == num_rows - 1:
            # Get the row index of the 1 entry in the column
            row_idx = col[col == 1].index[0]
            basis.append((row_idx, col_name))

    return basis


def choose_entering_variable(tableau):
    """
    Choose the entering variable for the pivoting step.

    Parameters:
    - tableau (pandas DataFrame): The current tableau of the LP problem.

    Returns:
    - str: The column name of the entering variable, or None if the solution is optimal.
    """
    # Objective function row is the first row
    obj_row = tableau.iloc[0, :-1]  # Exclude the RHS value

    # Identify the most negative coefficient
    min_coeff = obj_row.min()

    # If all coefficients are non-negative, the solution is optimal
    if min_coeff >= 0:
        return None

    # Return the column name of the most negative coefficient
    return obj_row.idxmin()


def choose_departing_variable(tableau, entering_col):
    """
    Choose the departing variable for the pivoting step.

    Parameters:
    - tableau (pandas DataFrame): The current tableau of the LP problem.
    - entering_col (str): The column name of the entering variable.

    Returns:
    - int: The index of the departing variable/row.
    """
    # Only compute ratios for rows where the value in the entering column is positive
    positive_rows = tableau[entering_col] > 0
    positive_ratios = tableau.loc[positive_rows, "RHS"] / tableau.loc[positive_rows, entering_col]

    # Return the row index of the smallest ratio
    return positive_ratios.idxmin()


def pivot(tableau, entering_col, departing_row):
    """
    Perform the pivot operation to update the tableau.

    Parameters:
    - tableau (pandas DataFrame): The current tableau of the LP problem.
    - entering_col (str): The column name of the entering variable.
    - departing_row (int): The index of the departing variable/row.

    Returns:
    - pandas DataFrame: The updated tableau.
    """
    # Create a copy of the tableau to avoid modifying the original
    new_tableau = tableau.copy()

    # Scale the departing row
    pivot_value = tableau.loc[departing_row, entering_col]
    new_tableau.loc[departing_row] /= pivot_value

    # Update the other rows
    for i in tableau.index:
        # Skip the departing row
        if i != departing_row:
            factor = tableau.loc[i, entering_col]
            new_tableau.loc[i] -= factor * new_tableau.loc[departing_row]

    return new_tableau


def extract_solution(tableau):
    """
    Extract the solution (values of decision variables) from the final tableau.

    Parameters:
    - tableau (pandas DataFrame): The final tableau of the LP problem.

    Returns:
    - solution (dict): A dictionary where keys are variable names (e.g., "x1") and values are the corresponding values.
    """
    basis_vars = get_basis_variables(tableau)
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


def get_constraints_str(objective_coeffs, objective_constant, constraints):
    num_coeffs = len(objective_coeffs)

    # Create a new pretty table
    pt = PrettyTable()

    # Set the column names
    header = ["Coeff" + str(i + 1) for i in range(num_coeffs)] + ["Relation", "RHS"]
    pt.field_names = header

    # Adding the objective coefficients row
    obj_row = ["{:.2f}".format(val) for val in objective_coeffs] + [
        "Obj. Func.",
        "{:.2f}".format(objective_constant),
    ]
    pt.add_row(obj_row)

    # Rows with dynamic width for constraints
    for coeffs, relation, rhs in constraints:
        formatted_row = ["{:.2f}".format(val) for val in coeffs] + [relation, "{:.2f}".format(rhs)]
        pt.add_row(formatted_row)

    return str(pt)


def get_tableau_str(
    tableau, show_basic_vars=True, rounding_accuracy=2, display_basicsolution=False
):
    """
    Display the tableau with the basis variables using prettytable.

    Parameters:
    - tableau (pandas DataFrame): The current tableau of the LP problem.
    - show_basic_vars (bool): Whether to display basic variable columns.
    - rounding_accuracy (int): Number of decimal places to round the values in the tableau.
    - display_basicsolution (bool): Whether to display the basic solution.

    Returns:
    - str: A string representation of the tableau with basis variables.
    """
    # Getting the basis variables (row, col) pairs
    basis_vars = get_basis_variables(tableau)
    basis_rows = {row: col for row, col in basis_vars}

    rows, cols = tableau.shape

    # Filter columns based on whether basic variables should be displayed
    if show_basic_vars:
        valid_cols = set(tableau.columns)
    else:
        valid_cols = set(tableau.columns) - set(col for _, col in basis_vars)

    # Create a new pretty table
    pt = PrettyTable()

    # Set the column names
    header = ["Basis"] + [col for col in tableau.columns if col in valid_cols]
    pt.field_names = header

    # Add rows to the table
    for i in range(rows):
        basis_col = basis_rows.get(i, "-")  # Using "-" for non-existent basis
        formatted_row = [basis_col] + [
            round(tableau.iloc[i][col], rounding_accuracy)
            for col in tableau.columns
            if col in valid_cols
        ]
        pt.add_row(formatted_row)

    tableau_str = str(pt)

    if display_basicsolution:
        solution_str = get_solution_str(extract_solution(tableau))
        tableau_str += f"\nBasic solution: {solution_str}"

    return tableau_str


def get_solution_str(solution, show_non_basic_vars=True, rounding_accuracy=2):
    """
    Display the solution in the desired format with underscores for non-zero values.

    Parameters:
    - solution (dict): The solution dictionary.
    - show_non_basic_vars (bool): Whether to display non-basic variables.
    - rounding_accuracy (int): Number of decimal places to round the solution values.

    Returns:
    - str: Formatted solution string.
    """
    # Filter out non-basic variables if requested
    if not show_non_basic_vars:
        solution = {k: v for k, v in solution.items() if v != 0}

    # Extract and format the keys and values based on the keys in the filtered solution dictionary
    keys = list(solution.keys())
    values = [round(solution[key], rounding_accuracy) for key in keys]

    # Format values, adding underscores to non-zero values
    formatted_values = [f"\033[4m{value}\033[0m" if value != 0 else str(value) for value in values]

    # Combine variable names and their values for display
    return f"({' '.join(keys)}) = ({' '.join(formatted_values)})"


if __name__ == "__main__":
    # read from file
    in_file = "in.txt"
    objective_type, objective_coeffs, objective_constant, constraints = import_from_file(in_file)
    inequalities = [inequality for _, inequality, _ in constraints]
    print(chr(27) + "[2J")
    print("\n*** START ***")
    print(f"Read from file: {in_file}")
    print(f"Objective type: {objective_type.upper()}")
    print(get_constraints_str(objective_coeffs, objective_constant, constraints))

    # setup initial tableau
    tableau, tableau_d = setup_initial_tableau(
        objective_type, objective_coeffs, objective_constant, constraints
    )
    print("\nInitial Tableau:")
    print(get_tableau_str(tableau, display_basicsolution=True))
    basis_vars = get_basis_variables(tableau)

    # simplex algorithm
    print("\n*** START SIMPLEX ***")
    optimality = check_for_optimality(tableau)
    while not optimality:
        entering_variable = choose_entering_variable(tableau)
        departing_variable = choose_departing_variable(tableau, entering_variable)
        tableau = pivot(tableau, entering_variable, departing_variable)
        print(f"\nEntering variable: {entering_variable}")
        print(
            f"Departing variable: {next((col for row, col in basis_vars if row == departing_variable), None)}\n"
        )
        print(get_tableau_str(tableau, display_basicsolution=True))
        basis_vars = get_basis_variables(tableau)

        optimality = check_for_optimality(tableau)

    print("\nSolution is optimal.")

    solution = extract_solution(tableau)
    print(f"Solution: {get_solution_str(solution)}")

    print("\n*** END ***")
