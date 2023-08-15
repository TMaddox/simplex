import numpy as np


def import_from_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

        # Get the objective type
        objective_type = lines[0].strip().lower()

        # Get the coefficients of the objective function
        # Using eval to convert fractions (or any valid expression) into float
        objective_coeffs = [eval(coeff) for coeff in lines[1].split()]

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

    return objective_type, objective_coeffs, constraints

    return objective_type, objective_coeffs, constraints


def setup_initial_tableau(objective_coeffs, constraints):
    # Number of original variables (including x0)
    num_vars = len(objective_coeffs) + 1

    # Number of constraints (and hence potential slack variables)
    num_constraints = len(constraints)

    # Counting the total number of slack variables (excluding those for '=' constraints)
    num_slack_vars = sum(1 for _, relation, _ in constraints if relation != "=")

    # Initialize the tableau with zeros
    tableau = np.zeros((num_constraints + 1, num_vars + num_slack_vars + 1))

    # Set the objective function coefficients
    tableau[0, 0] = 1  # Coefficient for x0
    tableau[0, 1:num_vars] = -np.array(objective_coeffs)  # Coefficients for x1, x2, ...

    # Set the coefficients of the slack variables and right-hand side values
    for i, (coeffs, relation, rhs) in enumerate(constraints, 1):
        # Coefficients of the original variables
        tableau[i, :num_vars] = [0] + coeffs  # 1 for x0 and then the coefficients for x1, x2, ...

        # Coefficient for the slack variable (1 or -1 based on the relation)
        if relation == "<=":
            tableau[i, num_vars + i - 1] = 1
        elif relation == ">=":
            tableau[i, num_vars + i - 1] = -1
        # If it's '=', we don't need to add a slack variable

        # Right-hand side value
        tableau[i, -1] = rhs

    return tableau


def check_for_optimality(tableau):
    """
    Check if the current tableau represents an optimal solution.

    Parameters:
    - tableau (numpy array): The current tableau of the LP problem.

    Returns:
    - bool: True if the solution is optimal, False otherwise.
    """
    # The solution is optimal if all coefficients in the last row (objective function row) are non-negative
    return np.all(tableau[0, :-1] >= 0)


def get_basis_variables(tableau):
    """
    Get the row and column indices of the basic variables in the current tableau.

    Parameters:
    - tableau (numpy array): The current tableau of the LP problem.

    Returns:
    - basis (list of tuple): Each tuple contains the row and column indices of a basic variable.
    """
    num_rows, num_cols = tableau.shape
    basis = []

    # Iterate through columns (excluding the RHS column)
    for j in range(num_cols - 1):
        col = tableau[:, j]

        # Check if the column has exactly one entry of 1 and all others are 0
        if np.sum(col == 1) == 1 and np.sum(col == 0) == num_rows - 1:
            # Get the row index of the 1 entry in the column
            row_idx = np.where(col == 1)[0][0]
            basis.append((row_idx, j))

    return basis


def choose_entering_variable(tableau):
    """
    Choose the entering variable for the pivoting step.

    Parameters:
    - tableau (numpy array): The current tableau of the LP problem.

    Returns:
    - int: The index of the entering variable, or None if the solution is optimal.
    """
    # Objective function row is the first row
    obj_row = tableau[0, :-1]  # Exclude the RHS value

    # Identify the most negative coefficient
    min_coeff = np.min(obj_row)

    # If all coefficients are non-negative, the solution is optimal
    if min_coeff >= 0:
        return None

    # Return the index of the most negative coefficient
    return np.argmin(obj_row)


def choose_departing_variable(tableau, entering_col):
    """
    Choose the departing variable for the pivoting step.

    Parameters:
    - tableau (numpy array): The current tableau of the LP problem.
    - entering_col (int): The index of the entering variable/column.

    Returns:
    - int: The index of the departing variable/row.
    """
    # Get the number of rows in the tableau
    num_rows = tableau.shape[0]

    # Initialize ratios with infinity (so they don't interfere with the minimum)
    ratios = np.full(num_rows, np.inf)

    # Only compute ratios for rows where the value in the entering column is positive
    positive_rows = tableau[:, entering_col] > 0
    ratios[positive_rows] = tableau[positive_rows, -1] / tableau[positive_rows, entering_col]

    # Return the row index of the smallest ratio
    return np.argmin(ratios)


def pivot(tableau, entering_col, departing_row):
    """
    Perform the pivot operation to update the tableau.

    Parameters:
    - tableau (numpy array): The current tableau of the LP problem.
    - entering_col (int): The index of the entering variable/column.
    - departing_row (int): The index of the departing variable/row.

    Returns:
    - numpy array: The updated tableau.
    """
    # Create a copy of the tableau to avoid modifying the original
    new_tableau = tableau.copy()

    # Scale the departing row
    pivot_value = tableau[departing_row, entering_col]
    new_tableau[departing_row, :] /= pivot_value

    # Update the other rows
    for i in range(tableau.shape[0]):
        # Skip the departing row
        if i != departing_row:
            factor = tableau[i, entering_col]
            new_tableau[i, :] -= factor * new_tableau[departing_row, :]

    return new_tableau


def extract_solution(tableau, rounding_accuracy=2):
    """
    Extract the solution (values of decision variables) from the final tableau.

    Parameters:
    - tableau (numpy array): The final tableau of the LP problem.
    - rounding_accuracy (int): Number of decimal places to round the solution values.

    Returns:
    - solution (dict): A dictionary where keys are variable names (e.g., "x1") and values are the corresponding values.
    """
    basis_vars = get_basis_variables(tableau)
    solution = {}

    # For each basic variable, fetch its value
    for row, col in basis_vars:
        solution[f"x{col}"] = round(tableau[row, -1], rounding_accuracy)

    return solution


def display_tableau(tableau):
    """
    Display the tableau with the basis variables.

    Parameters:
    - tableau (numpy array): The current tableau of the LP problem.

    Returns:
    - str: A string representation of the tableau with basis variables.
    """
    # Getting the basis variables (row, col) pairs
    basis_vars = get_basis_variables(tableau)

    # Extracting just the column values to use for naming
    basis_rows = {row: col for row, col in basis_vars}

    rows, cols = tableau.shape
    table_str = ""

    # Determine the width for each column
    header = ["Basis"] + ["x" + str(i) for i in range(cols - 1)] + ["RHS"]
    max_lengths = [
        max(len(header[i]), max(len("{:.2f}".format(row[i - 1])) for row in tableau))
        for i in range(1, cols + 1)
    ]

    # Add width for Basis column
    max_lengths = [max(len(header[0]), 5)] + max_lengths

    # Header with dynamic width
    table_str += " | ".join([header[i].rjust(max_lengths[i]) for i in range(cols + 1)]) + "\n"
    table_str += "-" * (sum(max_lengths) + 3 * cols) + "\n"  # Separator

    # Rows with dynamic width (with basis)
    for i, row in enumerate(tableau):
        basis_col = [("x" + str(basis_rows[i])).ljust(max_lengths[0])]
        formatted_row = basis_col + [
            "{:.2f}".format(val).rjust(max_lengths[j + 1]) for j, val in enumerate(row)
        ]
        table_str += " | ".join(formatted_row) + "\n"

    return table_str


def display_constraints_with_objective(objective_coeffs, constraints):
    num_coeffs = len(objective_coeffs)
    table_str = ""

    # Determine the width for each column
    header = ["Coeff" + str(i + 1) for i in range(num_coeffs)] + [
        "Relation",
        "RHS",
    ]
    max_lengths = [
        max(
            len(header[i]),
            max(len("{:.2f}".format(constraint[0][i])) for constraint in constraints),
        )
        for i in range(num_coeffs)
    ]
    # Setting fixed widths for 'Relation' and 'RHS' columns
    max_lengths += [10, 5]

    # Header with dynamic width
    table_str += (
        " | ".join([header[i].center(max_lengths[i]) for i in range(num_coeffs + 2)]) + "\n"
    )
    table_str += "-" * (sum(max_lengths) + 3 * (num_coeffs + 1)) + "\n"  # Separator

    # Adding the objective coefficients row
    obj_row = [
        "{:.2f}".format(val).rjust(max_lengths[i]) for i, val in enumerate(objective_coeffs)
    ] + ["Obj. Func.".center(max_lengths[-2]), "".center(max_lengths[-1])]
    table_str += " | ".join(obj_row) + "\n"

    # Rows with dynamic width for constraints
    for coeffs, relation, rhs in constraints:
        formatted_row = ["{:.2f}".format(val).rjust(max_lengths[i]) for i, val in enumerate(coeffs)]
        formatted_row += [
            relation.center(max_lengths[-2]),
            "{:.2f}".format(rhs).rjust(max_lengths[-1]),
        ]
        table_str += " | ".join(formatted_row) + "\n"

    return table_str


# read from file
in_file = "in.txt"
objective_type, objective_coeffs, constraints = import_from_file(in_file)
print("\n*** START ***")
print(f"Read from file: {in_file}")
print(f"Objective type: {objective_type.upper()}")
print(display_constraints_with_objective(objective_coeffs, constraints))

# setup initial tableau
tableau = setup_initial_tableau(objective_coeffs, constraints)
print("Initial Tableau:")
print(display_tableau(tableau))
basis_vars = get_basis_variables(tableau)
basis = ", ".join([f"x{col}" for _, col in basis_vars])
print(f"Basic variables: {basis}\n")

# simplex algorithm
optimality = check_for_optimality(tableau)
while not optimality:
    entering_variable = choose_entering_variable(tableau)
    departing_variable = choose_departing_variable(tableau, entering_variable)
    tableau = pivot(tableau, entering_variable, departing_variable)
    print(f"Entering variable: x{entering_variable}")
    print(f"Departing variable: x{basis[departing_variable]}")
    print(display_tableau(tableau))
    basis_vars = get_basis_variables(tableau)
    basis = ", ".join([f"x{col}" for _, col in basis_vars])
    print(f"Basic variables: {basis}\n")

    optimality = check_for_optimality(tableau)

print("Solution is optimal.")

solution = extract_solution(tableau)
print(f"Solution: {solution}")

print("\n*** END ***")
