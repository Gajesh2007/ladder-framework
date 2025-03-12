"""
Test script for the Variant Generator using mocks
This version doesn't require PyTorch or actual language models
"""

import sys
import os
import logging
import json

# Add the parent directory to the Python path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import config without importing the variant generator
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)

class MockVariantGenerator:
    """Mock implementation of the variant generator for testing"""
    
    def __init__(self, model_path=None, tokenizer_path=None, device="cpu"):
        self.model_path = model_path or "mock_model"
        self.tokenizer_path = tokenizer_path or "mock_tokenizer"
        self.device = device
        self.config = Config.VGSConfig
        print(f"Initialized MockVariantGenerator with {self.model_path}")
    
    def generate_variants(self, problem, num_variants=1, difficulty="easier"):
        print(f"Generating {num_variants} {difficulty} variants for: {problem}")
        if difficulty == "easier":
            return [f"Simpler variant of {problem} #{i+1}" for i in range(num_variants)]
        else:
            return [f"Equivalent variant of {problem} #{i+1}" for i in range(num_variants)]
    
    def generate_with_temperature_cycling(self, problem, num_variants=1, difficulty="easier"):
        print(f"Temperature cycling for {num_variants} variants")
        temps = self.config.TEMPERATURE_RANGE
        print(f"Using temperature range: {temps}")
        return self.generate_variants(problem, num_variants, difficulty)
    
    def generate_with_persona_prompting(self, problem, num_variants=1, difficulty="easier"):
        print(f"Persona prompting for {num_variants} variants")
        print(f"Using {self.config.NUM_PERSONAS} different mathematical personas")
        return self.generate_variants(problem, num_variants, difficulty)
    
    def generate_variant_tree(self, root_problem, max_depth=None, branching_factor=None):
        max_depth = max_depth or self.config.MAX_DEPTH
        branching_factor = branching_factor or self.config.BRANCHING_FACTOR
        
        print(f"Generating variant tree for: {root_problem}")
        print(f"Max depth: {max_depth}, branching factor: {branching_factor}")
        
        # Create a tree with the specified structure
        tree = {"problem": root_problem, "children": []}
        
        def build_tree(node, current_depth):
            if current_depth >= max_depth:
                return
                
            for i in range(branching_factor):
                child_problem = f"Level {current_depth+1} variant #{i+1} of {node['problem']}"
                child = {"problem": child_problem, "children": []}
                node["children"].append(child)
                build_tree(child, current_depth + 1)
        
        build_tree(tree, 0)
        return tree

def test_variant_generation():
    """Test basic variant generation functionality"""
    print("\nTesting variant generation...")
    
    # Create the mock generator
    generator = MockVariantGenerator(model_path="meta-llama/Llama-2-7b")
    
    # Test generating variants
    problems = [
        "Integrate x^2 * sin(x) dx",
        "Integrate e^x / (1 + e^x) dx",
        "Integrate ln(x) / x dx"
    ]
    
    for problem in problems:
        print(f"\nTesting with problem: {problem}")
        
        # Basic variant generation
        variants = generator.generate_variants(problem, num_variants=2)
        print(f"Generated variants: {variants}")
        
        # Temperature cycling
        variants = generator.generate_with_temperature_cycling(problem, num_variants=2)
        print(f"Temperature cycling variants: {variants}")
        
        # Persona prompting
        variants = generator.generate_with_persona_prompting(problem, num_variants=2)
        print(f"Persona prompting variants: {variants}")

def test_variant_tree_generation():
    """Test variant tree generation"""
    print("\nTesting variant tree generation...")
    
    # Create the mock generator
    generator = MockVariantGenerator()
    
    # Test generating a variant tree
    problem = "Integrate x^2 * sin(x) dx"
    tree = generator.generate_variant_tree(problem, max_depth=2, branching_factor=2)
    
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
    expected_count = 1 + 2 + 4  # root + level 1 + level 2
    
    print(f"\nTotal nodes in tree: {node_count}")
    print(f"Expected nodes for depth=2, branching=2: {expected_count}")
    
    assert node_count == expected_count, f"Node count {node_count} doesn't match expected {expected_count}"
    print("Tree structure verified correctly!")

def test_variant_api_alignment():
    """Test that the mock API aligns with the expected real API"""
    print("\nVerifying mock API alignment...")
    
    # Define expected methods for the real VariantGenerator
    expected_methods = [
        "generate_variants",
        "generate_with_temperature_cycling",
        "generate_with_persona_prompting", 
        "generate_variant_tree"
    ]
    
    # Check that our mock implements all these methods
    for method_name in expected_methods:
        if hasattr(MockVariantGenerator, method_name):
            print(f"✓ Mock implements {method_name}")
        else:
            print(f"✗ Mock is missing {method_name}")
    
    # Demonstrate how the methods would be used in production
    print("\nIn production, a variant tree would be used to:")
    print("1. Generate a hierarchical set of progressively simpler problems")
    print("2. Feed these variants into the RL subsystem for training")
    print("3. Create a natural curriculum based on problem difficulty")

if __name__ == "__main__":
    print("VARIANT GENERATOR TESTS (MOCK VERSION)")
    print("=" * 40)
    
    test_variant_generation()
    test_variant_tree_generation()
    test_variant_api_alignment()
    
    print("\nTests completed successfully")
