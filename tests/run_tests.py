#!/usr/bin/env python3
"""
LADDER Framework Test Runner

This script runs all test scripts in the tests directory to validate components of the LADDER framework.
"""

import os
import sys
import importlib
import subprocess
from typing import List, Tuple

# Set up paths
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, PROJECT_ROOT)

def print_header(title: str, width: int = 80):
    """Print a formatted header"""
    print("\n" + "=" * width)
    print(f"{title.center(width)}")
    print("=" * width + "\n")

def list_test_modules() -> List[str]:
    """List all test modules in the tests directory"""
    test_files = [
        f[:-3] for f in os.listdir(TESTS_DIR) 
        if f.startswith("test_") and f.endswith(".py") and f != "test_generator.py"
    ]
    
    # Include the mock generator test instead of the real one that requires PyTorch
    if "test_generator_mock.py" in os.listdir(TESTS_DIR):
        test_files.append("test_generator_mock")
    
    return sorted(test_files)

def run_module_tests(module_name: str) -> Tuple[bool, str]:
    """Run tests in a specific module"""
    try:
        print_header(f"Running {module_name}")
        
        # Run the module as a script to capture all output
        result = subprocess.run(
            [sys.executable, os.path.join(TESTS_DIR, f"{module_name}.py")],
            capture_output=True,
            text=True
        )
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print("ERRORS:")
            print(result.stderr)
        
        return result.returncode == 0, result.stdout
    except Exception as e:
        print(f"Error running {module_name}: {e}")
        return False, str(e)

def check_environment():
    """Check if we have the required libraries for testing"""
    print("Checking environment...")
    
    min_requirements = ["numpy", "sympy"]
    ml_requirements = ["torch", "transformers"]
    
    installed = []
    missing = []
    
    # Check minimal requirements
    for lib in min_requirements:
        try:
            importlib.import_module(lib)
            installed.append(lib)
        except ImportError:
            missing.append(lib)
    
    # Check ML requirements
    have_ml = True
    for lib in ml_requirements:
        try:
            importlib.import_module(lib)
            installed.append(lib)
        except ImportError:
            missing.append(lib)
            have_ml = False
    
    print(f"Installed libraries: {', '.join(installed)}")
    if missing:
        print(f"Missing libraries: {', '.join(missing)}")
    
    print(f"Running in {'full ML' if have_ml else 'limited'} test mode")
    print(f"Using {'real' if have_ml else 'mock'} implementations for ML components")
    
    return have_ml

def run_all_tests():
    """Run all test modules"""
    print_header("LADDER FRAMEWORK TEST SUITE", 60)
    
    # Check environment
    have_ml = check_environment()
    
    # Get list of test modules
    test_modules = list_test_modules()
    print(f"Found {len(test_modules)} test modules: {', '.join(test_modules)}\n")
    
    # Run each test module
    results = []
    for module in test_modules:
        success, _ = run_module_tests(module)
        results.append((module, success))
    
    # Print summary
    print_header("TEST SUMMARY", 60)
    for module, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{module}: {status}")
    
    # Overall status
    all_passed = all(success for _, success in results)
    print("\nOverall result:", "PASSED" if all_passed else "FAILED")
    
    if not have_ml:
        print("\nNote: Tests were run in limited mode without ML libraries.")
        print("Install PyTorch and Transformers for full testing capabilities.")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
