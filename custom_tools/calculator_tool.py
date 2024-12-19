from sympy import sympify

def calculator_tool(expression):
    try:
        # Safely parse and evaluate the expression
        result = sympify(expression).evalf()
        return result
    except Exception as e:
        return f"Error: {str(e)}"
