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


def display_tableau(tableau):
    rows, cols = tableau.shape
    table_str = ""

    # Adjusting for x0 representation for the objective function
    header = ["x" + str(i) for i in range(cols - 1)] + ["RHS"]
    max_lengths = [
        max(
            len(header[i]),
            max(len("{:.2f}".format(row[i])) for row in tableau),
        )
        for i in range(cols)
    ]

    # Header with dynamic width
    table_str += " | ".join([header[i].rjust(max_lengths[i]) for i in range(cols)]) + "\n"
    table_str += "-" * (sum(max_lengths) + 3 * (cols - 1)) + "\n"  # Separator

    # Rows with dynamic width
    for row in tableau:
        formatted_row = ["{:.2f}".format(val).rjust(max_lengths[i]) for i, val in enumerate(row)]
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


in_file = "in.txt"
objective_type, objective_coeffs, constraints = import_from_file(in_file)
print("\n*** START ***")
print(f"Read from file: {in_file}")
print(f"Objective type: {objective_type.upper()}\n")
print(display_constraints_with_objective(objective_coeffs, constraints))
print("\nSetting up initial tableau...")
tableau = setup_initial_tableau(objective_coeffs, constraints)
print("\nInitial Tableau:")
print(display_tableau(tableau))
print("\n*** END ***")
