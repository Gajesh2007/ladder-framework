"""
Test script for the Integration Verifier
"""

import sys
import os
import logging
import sympy as sp
import numpy as np

# Add the parent directory to the Python path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verification.verifier import IntegrationVerifier

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_simple_integrals():
    """Test basic polynomial integration verification"""
    verifier = IntegrationVerifier()
    
    test_cases = [
        # Polynomials
        ("x^2", "x^3/3 + C", True),
        ("x^3", "x^4/4 + C", True),
        ("2*x + 3", "x^2 + 3*x + C", True),
        ("x^2", "x^2 + C", False),  # Incorrect
    ]
    
    print("\nTesting simple polynomial integrals...")
    for problem, solution, expected in test_cases:
        result = verifier.verify_solution(problem, solution)
        status = "✓" if result == expected else "✗"
        print(f"{status} ∫ {problem} dx = {solution} (Expected: {expected}, Got: {result})")

def fix_substitution_issue():
    """
    Fix the substitution issue in the verifier
    This is a demonstration of one approach to fixing the parsing issues
    """
    print("\nDemonstrating a fix for the substitution issues...")
    # Create a verifier and modify its substitution method
    verifier = IntegrationVerifier()
    
    # Define a patched version of _parse_expression
    def patched_parse_expression(expr_str):
        """Patched version that handles functions directly"""
        expr_str = expr_str.strip()
        
        # Replace common functions directly with their sympy equivalents
        expr_str = expr_str.replace("sin(", "sp.sin(")
        expr_str = expr_str.replace("cos(", "sp.cos(")
        expr_str = expr_str.replace("tan(", "sp.tan(")
        expr_str = expr_str.replace("e^", "sp.exp(")
        
        # Replace x^n with x**n for SymPy compatibility
        expr_str = expr_str.replace("^", "**")
        
        # Replace C (constant of integration) with a symbol
        expr_str = expr_str.replace(" + C", " + sp.Symbol('C')")
        
        print(f"Parsed '{expr_str}' as sympy expression")
        return sp.sympify(expr_str, locals={'x': sp.Symbol('x'), 'sp': sp})
    
    # Test the patched function
    try:
        expr = patched_parse_expression("sin(x)")
        print(f"Successfully parsed 'sin(x)' as {expr}")
        expr = patched_parse_expression("-cos(x) + C")
        print(f"Successfully parsed '-cos(x) + C' as {expr}")
    except Exception as e:
        print(f"Error: {e}")

def test_numerical_verification():
    """Test the numerical verification approach"""
    print("\nDemonstrating numerical verification...")
    
    # Define a function and its derivative using numpy directly
    def f(x):
        return np.sin(x)
    
    def df(x):
        return np.cos(x)
    
    # Sample points
    x_values = np.linspace(-5, 5, 10)
    
    # Calculate values
    f_values = f(x_values)
    df_values = df(x_values)
    
    # Calculate max difference
    max_diff = np.max(np.abs(df_values - f_values))
    print(f"Maximum difference between sin(x) and cos(x): {max_diff}")
    
    # This would fail since sin(x) is not the derivative of cos(x)
    # In a proper verification, we'd check if the difference is below a threshold
    threshold = 0.1
    is_verified = max_diff < threshold
    print(f"Verification result (should be False): {is_verified}")

if __name__ == "__main__":
    print("VERIFIER TESTS")
    print("=" * 40)
    
    test_simple_integrals()
    fix_substitution_issue()
    test_numerical_verification()
    
    print("\nTests completed")
