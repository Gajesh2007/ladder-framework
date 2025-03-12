"""
Verification Subsystem (VS)

Provides ground-truth verification of solution correctness through numerical evaluation.
For mathematical integration, uses numerical integration with symbolic preprocessing.
"""

import re
import time
import logging
import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Union, Optional, Callable

from config.config import Config

logger = logging.getLogger(__name__)

class VerificationError(Exception):
    """Exception raised for errors in the verification process."""
    pass

class IntegrationVerifier:
    """
    Verifies the correctness of integration solutions through numerical evaluation.
    
    Key Features:
    - Multi-point evaluation across domain [-10, 10]
    - Adaptive error handling for singularities and numerical instability
    - Support for both definite and indefinite integrals
    """
    
    def __init__(self, config=None):
        """
        Initialize the Integration Verifier.
        
        Args:
            config: Configuration object, defaults to the global Config
        """
        self.config = config or Config.VSConfig
        
        # Define symbolic variable for SymPy
        self.x = sp.Symbol('x')
        
        # Define substitution patterns for parsing
        self.substitutions = {
            # Common mathematical constants
            'pi': 'sp.pi',
            'e': 'sp.E',
            
            # Common mathematical functions
            'sin': 'sp.sin',
            'cos': 'sp.cos',
            'tan': 'sp.tan',
            'cot': 'sp.cot',
            'sec': 'sp.sec',
            'csc': 'sp.csc',
            'log': 'sp.log',
            'ln': 'sp.log',
            'exp': 'sp.exp',
            'sqrt': 'sp.sqrt',
            'arcsin': 'sp.asin',
            'arccos': 'sp.acos',
            'arctan': 'sp.atan',
            'sinh': 'sp.sinh',
            'cosh': 'sp.cosh',
            'tanh': 'sp.tanh',
        }
    
    def verify_solution(
        self, 
        problem: str, 
        solution: str, 
        detailed: bool = False
    ) -> Union[bool, Dict[str, Union[bool, str, float]]]:
        """
        Verify if a solution to an integration problem is correct.
        
        Args:
            problem: The integration problem statement
            solution: The proposed solution
            detailed: Whether to return detailed verification results
            
        Returns:
            If detailed=True, returns a dictionary with verification details.
            Otherwise, returns a boolean indicating whether the solution is correct.
        """
        try:
            # Parse the problem and solution
            integrand, limits = self._parse_problem(problem)
            proposed_answer = self._parse_solution(solution)
            
            if limits:
                # Definite integral
                correct, error_info = self._verify_definite_integral(
                    integrand, proposed_answer, limits
                )
            else:
                # Indefinite integral
                correct, error_info = self._verify_indefinite_integral(
                    integrand, proposed_answer
                )
            
            if detailed:
                return {
                    "correct": correct,
                    "error": error_info.get("error", ""),
                    "max_error": error_info.get("max_error", 0.0),
                    "integrand": str(integrand),
                    "parsed_solution": str(proposed_answer),
                    "limits": str(limits) if limits else "indefinite"
                }
            
            return correct
            
        except Exception as e:
            logger.error(f"Verification error: {str(e)}")
            
            if detailed:
                return {
                    "correct": False,
                    "error": str(e),
                    "max_error": float('inf'),
                    "integrand": str(integrand) if 'integrand' in locals() else "parsing error",
                    "parsed_solution": str(proposed_answer) if 'proposed_answer' in locals() else "parsing error",
                    "limits": str(limits) if 'limits' in locals() and limits else "indefinite"
                }
            
            return False
    
    def _parse_problem(self, problem: str) -> Tuple[sp.Expr, Optional[Tuple[float, float]]]:
        """
        Parse an integration problem into a SymPy expression and limits.
        
        Args:
            problem: The integration problem statement
            
        Returns:
            Tuple of (integrand expression, integration limits or None for indefinite integrals)
        """
        # Extract the integrand and limits using regex
        indefinite_pattern = r'∫\s*(.*?)\s*dx'
        definite_pattern = r'∫_\{(.*?)\}\^\{(.*?)\}\s*(.*?)\s*dx'
        alternative_definite_pattern = r'∫\s*\[([^,]*),\s*([^,]*)\]\s*(.*?)\s*dx'
        
        # Try matching definite integral patterns
        definite_match = re.search(definite_pattern, problem)
        if definite_match:
            lower_limit = self._parse_expression(definite_match.group(1))
            upper_limit = self._parse_expression(definite_match.group(2))
            integrand_str = definite_match.group(3)
            limits = (float(lower_limit), float(upper_limit))
        else:
            alternative_match = re.search(alternative_definite_pattern, problem)
            if alternative_match:
                lower_limit = self._parse_expression(alternative_match.group(1))
                upper_limit = self._parse_expression(alternative_match.group(2))
                integrand_str = alternative_match.group(3)
                limits = (float(lower_limit), float(upper_limit))
            else:
                # Try matching indefinite integral pattern
                indefinite_match = re.search(indefinite_pattern, problem)
                if indefinite_match:
                    integrand_str = indefinite_match.group(1)
                    limits = None
                else:
                    # If nothing matches, assume the entire problem is the integrand
                    integrand_str = problem
                    limits = None
        
        # Parse the integrand string into a SymPy expression
        integrand = self._parse_expression(integrand_str)
        
        return integrand, limits
    
    def _parse_solution(self, solution: str) -> sp.Expr:
        """
        Parse a solution string into a SymPy expression.
        
        Args:
            solution: The solution string
            
        Returns:
            SymPy expression representing the solution
        """
        # Extract the answer from the solution if it's wrapped in answer tags
        answer_pattern = r'<ANSWER>(.*?)</ANSWER>'
        match = re.search(answer_pattern, solution, re.DOTALL)
        
        if match:
            answer_str = match.group(1).strip()
        else:
            # If no tags, use the entire solution
            answer_str = solution.strip()
        
        # Parse the answer string into a SymPy expression
        return self._parse_expression(answer_str)
    
    def _parse_expression(self, expr_str: str) -> sp.Expr:
        """
        Parse a mathematical expression string into a SymPy expression.
        
        Args:
            expr_str: The expression string
            
        Returns:
            SymPy expression
        """
        # Clean the expression string
        expr_str = expr_str.strip()
        
        # Apply substitutions for mathematical functions and constants
        for pattern, replacement in self.substitutions.items():
            expr_str = re.sub(r'\b' + pattern + r'\b', replacement, expr_str)
        
        # Replace x^n with x**n for SymPy compatibility
        expr_str = re.sub(r'(\w+)\^(\w+|\(.*?\))', r'\1**\2', expr_str)
        
        try:
            # Parse the expression string
            expr = sp.sympify(expr_str, locals={'x': self.x})
            return expr
        except Exception as e:
            logger.error(f"Error parsing expression '{expr_str}': {e}")
            raise VerificationError(f"Failed to parse expression: {expr_str}")
    
    def _verify_indefinite_integral(
        self, 
        integrand: sp.Expr, 
        proposed_answer: sp.Expr
    ) -> Tuple[bool, Dict[str, Union[str, float]]]:
        """
        Verify an indefinite integral solution.
        
        Args:
            integrand: The integrand expression
            proposed_answer: The proposed solution expression
            
        Returns:
            Tuple of (correct boolean, error information dictionary)
        """
        # Take the derivative of the proposed answer
        try:
            derivative = sp.diff(proposed_answer, self.x)
            
            # Create numerical functions
            integrand_func = sp.lambdify(self.x, integrand, "numpy")
            derivative_func = sp.lambdify(self.x, derivative, "numpy")
            
            # Sample points in the domain
            points = self._sample_points()
            
            # Calculate the maximum error across sample points
            max_error, error_point = self._calculate_max_error(
                integrand_func, derivative_func, points
            )
            
            # Check if the error is within tolerance
            within_tolerance = max_error <= self.config.RELATIVE_TOLERANCE
            
            error_info = {
                "max_error": max_error,
                "error_point": error_point,
                "error": "" if within_tolerance else f"Maximum error ({max_error}) exceeds tolerance at x={error_point}"
            }
            
            return within_tolerance, error_info
            
        except Exception as e:
            return False, {"error": str(e), "max_error": float('inf')}
    
    def _verify_definite_integral(
        self, 
        integrand: sp.Expr, 
        proposed_answer: sp.Expr, 
        limits: Tuple[float, float]
    ) -> Tuple[bool, Dict[str, Union[str, float]]]:
        """
        Verify a definite integral solution.
        
        Args:
            integrand: The integrand expression
            proposed_answer: The proposed solution expression (should be a constant)
            limits: The integration limits (lower, upper)
            
        Returns:
            Tuple of (correct boolean, error information dictionary)
        """
        # Calculate the numerical value of the definite integral
        try:
            lower_limit, upper_limit = limits
            
            # Symbolic integration
            symbolic_result = sp.integrate(integrand, (self.x, lower_limit, upper_limit))
            symbolic_value = float(symbolic_result.evalf())
            
            # Numeric integration for verification
            integrand_func = sp.lambdify(self.x, integrand, "numpy")
            numeric_value = self._numeric_integrate(
                integrand_func, lower_limit, upper_limit
            )
            
            # Extract numerical value from proposed answer
            proposed_value = float(proposed_answer.evalf())
            
            # Calculate relative errors
            symbolic_error = abs((proposed_value - symbolic_value) / (symbolic_value + 1e-10))
            numeric_error = abs((proposed_value - numeric_value) / (numeric_value + 1e-10))
            
            # Use the minimum error (giving benefit of the doubt)
            min_error = min(symbolic_error, numeric_error)
            
            # Check if the error is within tolerance
            within_tolerance = min_error <= self.config.RELATIVE_TOLERANCE
            
            error_info = {
                "max_error": min_error,
                "symbolic_value": symbolic_value,
                "numeric_value": numeric_value,
                "proposed_value": proposed_value,
                "error": "" if within_tolerance else f"Error ({min_error}) exceeds tolerance"
            }
            
            return within_tolerance, error_info
            
        except Exception as e:
            return False, {"error": str(e), "max_error": float('inf')}
    
    def _sample_points(self) -> np.ndarray:
        """
        Generate sample points for numerical evaluation.
        
        Returns:
            Array of sample points
        """
        start, end = self.config.EVAL_DOMAIN
        num_points = self.config.NUM_EVAL_POINTS
        
        # Generate evenly spaced points in the domain
        points = np.linspace(start, end, num_points)
        
        # Add some random points for robustness
        random_points = np.random.uniform(start, end, num_points // 2)
        
        return np.concatenate([points, random_points])
    
    def _calculate_max_error(
        self, 
        func1: Callable, 
        func2: Callable, 
        points: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate the maximum relative error between two functions across sample points.
        
        Args:
            func1: First function
            func2: Second function
            points: Sample points
            
        Returns:
            Tuple of (maximum error, point with maximum error)
        """
        max_error = 0.0
        error_point = 0.0
        
        for point in points:
            try:
                # Calculate function values
                val1 = func1(point)
                val2 = func2(point)
                
                # Calculate relative error
                if abs(val1) < 1e-10 and abs(val2) < 1e-10:
                    # Both close to zero, consider them equal
                    error = 0.0
                else:
                    error = abs((val1 - val2) / (abs(val1) + 1e-10))
                
                # Update maximum error
                if error > max_error:
                    max_error = error
                    error_point = point
                    
            except (ZeroDivisionError, ValueError, TypeError, OverflowError):
                # Skip points with numerical issues
                continue
        
        return max_error, error_point
    
    def _numeric_integrate(
        self, 
        func: Callable, 
        lower: float, 
        upper: float, 
        max_retries: int = None
    ) -> float:
        """
        Perform numerical integration using adaptive quadrature.
        
        Args:
            func: The function to integrate
            lower: Lower integration limit
            upper: Upper integration limit
            max_retries: Maximum number of retry attempts
            
        Returns:
            Numerical value of the integral
        """
        if max_retries is None:
            max_retries = self.config.MAX_RETRY_COUNT
        
        # Handle infinity in limits
        if np.isinf(lower) or np.isinf(upper):
            return self._handle_infinite_limits(func, lower, upper)
        
        # Set timeout for integration
        timeout = self.config.TIMEOUT_THRESHOLD
        result = None
        
        # Try integration with increasing number of sample points
        samples = [100, 500, 1000, 5000]
        
        for retry in range(max_retries):
            try:
                start_time = time.time()
                
                # Use trapezoidal rule with increasing number of sample points
                num_samples = samples[min(retry, len(samples) - 1)]
                dx = (upper - lower) / num_samples
                x = np.linspace(lower, upper, num_samples + 1)
                y = np.array([func(xi) for xi in x])
                result = np.trapz(y, x)
                
                # Check if integration completed within timeout
                if time.time() - start_time > timeout:
                    logger.warning(f"Integration timed out on retry {retry}")
                    continue
                
                # Check for NaN or Inf
                if np.isnan(result) or np.isinf(result):
                    logger.warning(f"Integration resulted in {result} on retry {retry}")
                    continue
                
                return result
                
            except Exception as e:
                logger.warning(f"Integration error on retry {retry}: {e}")
        
        # If all retries failed, raise an exception
        raise VerificationError("Failed to numerically integrate the function")
    
    def _handle_infinite_limits(
        self, 
        func: Callable, 
        lower: float, 
        upper: float
    ) -> float:
        """
        Handle integration with infinite limits using variable substitution.
        
        Args:
            func: The function to integrate
            lower: Lower integration limit
            upper: Upper integration limit
            
        Returns:
            Numerical value of the integral
        """
        if np.isinf(lower) and np.isinf(upper):
            # Double infinite integral: split into two parts
            # ∫_{-∞}^{∞} f(x) dx = ∫_{-∞}^{0} f(x) dx + ∫_{0}^{∞} f(x) dx
            neg_part = self._handle_infinite_limits(func, lower, 0)
            pos_part = self._handle_infinite_limits(func, 0, upper)
            return neg_part + pos_part
            
        elif np.isinf(upper):
            # Upper limit is infinity: substitute x = 1/t
            # ∫_{a}^{∞} f(x) dx = ∫_{0}^{1/a} f(1/t) * (1/t^2) dt
            if lower == 0:
                # Avoid division by zero in transformation
                lower = 1e-10
            
            def transformed_func(t):
                if t == 0:
                    return 0  # Avoid division by zero
                x = 1/t
                return func(x) / (t**2)
            
            return self._numeric_integrate(
                transformed_func, 0, 1/lower if lower != 0 else 1e10
            )
            
        elif np.isinf(lower):
            # Lower limit is -infinity: substitute x = -1/t
            # ∫_{-∞}^{b} f(x) dx = ∫_{0}^{-1/b} f(-1/t) * (1/t^2) dt
            if upper == 0:
                # Avoid division by zero in transformation
                upper = -1e-10
            
            def transformed_func(t):
                if t == 0:
                    return 0  # Avoid division by zero
                x = -1/t
                return func(x) / (t**2)
            
            return self._numeric_integrate(
                transformed_func, 0, -1/upper if upper != 0 else 1e10
            )
        
        # Should never reach here
        raise VerificationError("Invalid infinite limits handling")
