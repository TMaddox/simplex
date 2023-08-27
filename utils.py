import ast


def safe_eval(node_or_string):
    """
    Safely evaluate an arithmetic expression using the AST.
    """
    if isinstance(node_or_string, str):
        node = ast.parse(node_or_string, mode="eval").body
    else:
        node = node_or_string

    if isinstance(node, ast.Expression):
        return safe_eval(node.body)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left = safe_eval(node.left)
        right = safe_eval(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            return left / right
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):  # Handle negative numbers
        operand = safe_eval(node.operand)
        return -operand
    elif isinstance(node, ast.Num):
        return node.n
    else:
        raise ValueError("Unsupported operation. Only basic arithmetic operations are allowed.")


def to_float(x):
    """
    Converts a given input to a float or a list of floats.

    Parameters:
    - x (int, float, str, or list): The input to be converted.

    Returns:
    - float or list: The converted input. If the input is a list, each item in the list is converted.

    Raises:
    - ValueError: If a string input cannot be converted to a float.
    - TypeError: If an unsupported type is provided.
    """
    if isinstance(x, int):
        return float(x)
    elif isinstance(x, float):
        return x
    elif isinstance(x, str):
        try:
            return safe_eval(ast.parse(x, mode="eval"))
        except (ValueError, SyntaxError):
            raise ValueError(f"The string '{x}' cannot be converted to float.")
    elif isinstance(x, list):
        return [to_float(item) for item in x]
    else:
        raise TypeError(f"Unsupported type {type(x)} for conversion to float.")
