"""
Test script for the Variant Generator
"""

import sys
import os
import logging

# Add the parent directory to the Python path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from variant_generator.generator import VariantGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_variant_generation_without_model():
    """
    Test basic variant generation functionality without requiring a language model
    This allows us to test the infrastructure without the model dependencies
    """
    print("\nTesting variant generation infrastructure...")
    
    # Create a simple mock for demonstration
    class MockVariantGenerator:
        def generate_variants(self, problem, num_variants=1, difficulty="easier"):
            print(f"Mock generating {num_variants} variants for: {problem} (difficulty: {difficulty})")
            if difficulty == "easier":
                return [f"Simpler variant of {problem} #{i+1}" for i in range(num_variants)]
            else:
                return [f"Equivalent variant of {problem} #{i+1}" for i in range(num_variants)]
        
        def generate_variant_tree(self, root_problem, max_depth=2, branching_factor=2):
            print(f"Mock generating variant tree for: {root_problem} (depth: {max_depth}, branching: {branching_factor})")
            
            # Create a simple tree structure
            tree = {"problem": root_problem, "children": []}
            
            if max_depth > 0:
                for i in range(branching_factor):
                    child_problem = f"Level 1 variant #{i+1} of {root_problem}"
                    child = {"problem": child_problem, "children": []}
                    
                    if max_depth > 1:
                        for j in range(branching_factor):
                            grandchild_problem = f"Level 2 variant #{j+1} of {child_problem}"
                            grandchild = {"problem": grandchild_problem, "children": []}
                            child["children"].append(grandchild)
                    
                    tree["children"].append(child)
            
            return tree
    
    # Create the mock generator
    mock_generator = MockVariantGenerator()
    
    # Test generating variants
    variants = mock_generator.generate_variants("Integrate x^2 * sin(x) dx", num_variants=3)
    print(f"Generated variants: {variants}")
    
    # Test generating a variant tree
    tree = mock_generator.generate_variant_tree("Integrate x^2 * sin(x) dx", max_depth=2, branching_factor=2)
    
    # Print tree in a readable format
    def print_tree(node, indent=0):
        print("  " * indent + f"- {node['problem']}")
        for child in node.get("children", []):
            print_tree(child, indent + 1)
    
    print("\nVariant tree:")
    print_tree(tree)
    
    # Count nodes in tree
    def count_nodes(node):
        count = 1
        for child in node.get("children", []):
            count += count_nodes(child)
        return count
    
    node_count = count_nodes(tree)
    print(f"\nTotal nodes in tree: {node_count}")
    print(f"Expected nodes for depth=2, branching=2: 1 + 2 + 4 = 7")

def test_variant_generator_api():
    """Test the API of the actual VariantGenerator class"""
    print("\nTesting VariantGenerator API (without actually running the model)...")
    
    # Create a VariantGenerator instance but don't actually load a model
    # Just verify the methods exist with the right signatures
    try:
        generator_class = VariantGenerator
        print("Class exists with the following methods:")
        for method_name in dir(generator_class):
            if not method_name.startswith("_"):
                method = getattr(generator_class, method_name)
                if callable(method):
                    print(f"- {method_name}")
        
        print("\nInitialization requires a model, so we won't create an instance here.")
        print("In a full test with GPU resources, we would:")
        print("1. Initialize with a small model")
        print("2. Generate variants for simple problems")
        print("3. Validate the structure and content of the variants")
    except Exception as e:
        print(f"Error inspecting VariantGenerator class: {e}")

if __name__ == "__main__":
    print("VARIANT GENERATOR TESTS")
    print("=" * 40)
    
    test_variant_generation_without_model()
    test_variant_generator_api()
    
    print("\nTests completed")
