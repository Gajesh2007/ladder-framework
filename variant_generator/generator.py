"""
Variant Generation Subsystem (VGS)

Responsible for constructing tree-structured problem variants with progressively decreasing difficulty.
Includes temperature cycling and persona-based prompting to increase diversity.
"""

import random
import logging
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.config import Config

logger = logging.getLogger(__name__)

class VariantGenerator:
    """
    Generates progressively simpler variants of complex problems through recursive decomposition.
    
    Key Features:
    - Hierarchical variant tree with configurable depth and branching factor
    - Temperature cycling for diversity
    - Persona-based prompting for different mathematical perspectives
    - Quality filters for variant validation
    """
    
    def __init__(
        self, 
        model_path: str = Config.ModelConfig.BASE_MODEL_NAME,
        tokenizer_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the Variant Generator with a language model.
        
        Args:
            model_path: Path to the pretrained language model
            tokenizer_path: Path to the tokenizer (defaults to model_path if None)
            device: Device to run the model on ("cuda" or "cpu")
        """
        self.config = Config.VGSConfig
        self.device = device
        
        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.float16,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path else model_path
        )
        
        # Initialize mathematical transformation templates
        self.transformations = self._initialize_transformations()
        
        # Initialize mathematical personas
        self.personas = self._initialize_personas()
    
    def _initialize_transformations(self) -> List[str]:
        """
        Initialize mathematical transformation templates for variant generation.
        
        Returns:
            List of transformation templates
        """
        return [
            "Try replacing the integrand {integrand} with a simpler function.",
            "Try simplifying the integration bounds from {bounds} to a smaller interval.",
            "Try replacing the variable {variable} in the integral with a simpler expression.",
            "Try removing a term from the integrand {integrand}.",
            "Try converting this definite integral to an indefinite integral.",
            "Try breaking down the integral into smaller parts using linearity.",
            "Try using a specific substitution like u={suggestion} to simplify the integral.",
            "Try eliminating one of the more complex terms in the integrand.",
            "Try using a trigonometric identity to simplify {integrand}.",
            "Try replacing complex functions with their Taylor series approximations.",
            "Try using symmetry arguments to simplify the integral.",
            "Try changing from Cartesian to polar coordinates.",
            "Try replacing a complex function with a simpler one that has similar behavior.",
            "Try approximating {integrand} with a polynomial of lower degree.",
            "Try separating the integrand into simpler, additive components.",
            "Try replacing a complex coefficient with a simpler value.",
            "Try keeping only the dominant term in the integrand.",
            "Try removing constraints or conditions from the problem.",
            "Try expressing the integral in terms of a standard, named integral.",
            "Try simplifying the integrand using algebraic manipulations.",
            "Try using partial fractions to simplify the rational function.",
            "Try introducing a parameter that makes the integral easier.",
            "Try replacing complex numbers with real numbers.",
            "Try simplifying boundary conditions or initial conditions.",
            "Try replacing a complex constraint with a simpler one.",
        ]
    
    def _initialize_personas(self) -> List[Dict[str, str]]:
        """
        Initialize mathematical personas for diverse variant generation.
        
        Returns:
            List of persona dictionaries with name and description
        """
        return [
            {
                "name": "Calculus Teacher",
                "description": "As a calculus teacher, I create simpler examples to help students understand complex concepts step by step."
            },
            {
                "name": "Computational Mathematician",
                "description": "As a computational mathematician, I focus on transforming problems into forms that are more amenable to numerical methods."
            },
            {
                "name": "Algebraic Simplifier",
                "description": "As an algebraic simplifier, I look for ways to reduce the complexity of expressions through algebraic manipulations."
            },
            {
                "name": "Geometric Visualizer",
                "description": "As a geometric visualizer, I transform problems to exploit geometric properties like symmetry and visual intuition."
            },
            {
                "name": "Applied Integrator",
                "description": "As an applied integrator, I focus on techniques from practical applications, often using approximations and common patterns."
            },
        ]
    
    def generate_variant_tree(
        self, 
        root_problem: str, 
        max_depth: int = None, 
        branching_factor: int = None
    ) -> Dict[str, Any]:
        """
        Generate a tree of progressively simpler variants from a root problem.
        
        Args:
            root_problem: The original complex problem
            max_depth: Maximum depth of the variant tree (default: from config)
            branching_factor: Number of variants per node (default: from config)
            
        Returns:
            Dictionary representing the variant tree structure
        """
        if max_depth is None:
            max_depth = self.config.MAX_DEPTH
        
        if branching_factor is None:
            branching_factor = self.config.BRANCHING_FACTOR
        
        logger.info(f"Generating variant tree for problem: {root_problem[:50]}...")
        
        # Create root node
        tree = {
            "problem": root_problem,
            "depth": 0,
            "children": []
        }
        
        # Recursively generate variants
        self._generate_variants_recursive(tree, max_depth, branching_factor)
        
        return tree
    
    def _generate_variants_recursive(
        self, 
        node: Dict[str, Any], 
        max_depth: int, 
        branching_factor: int
    ) -> None:
        """
        Recursively generate variants for a node in the tree.
        
        Args:
            node: Current node in the variant tree
            max_depth: Maximum depth of the variant tree
            branching_factor: Number of variants per node
        """
        # Stop recursion if maximum depth is reached
        if node["depth"] >= max_depth:
            return
        
        # Generate variants for the current node
        variants = self._generate_variants_for_problem(
            node["problem"], 
            branching_factor
        )
        
        # Add variants as children
        for variant in variants:
            child_node = {
                "problem": variant,
                "depth": node["depth"] + 1,
                "children": []
            }
            node["children"].append(child_node)
            
            # Recursively generate variants for the child node
            self._generate_variants_recursive(child_node, max_depth, branching_factor)
    
    def _generate_variants_for_problem(
        self, 
        problem: str, 
        count: int
    ) -> List[str]:
        """
        Generate a specified number of variants for a problem.
        
        Args:
            problem: The problem to generate variants for
            count: Number of variants to generate
            
        Returns:
            List of generated variant problems
        """
        variants = []
        batch_size = min(count, self.config.BATCH_SIZE)
        num_batches = (count + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            # Calculate actual batch size for the current batch
            current_batch_size = min(batch_size, count - len(variants))
            
            # Generate batch of variants with temperature cycling
            batch_variants = self._generate_variant_batch(
                problem, 
                current_batch_size
            )
            
            # Filter and add variants
            for variant in batch_variants:
                # Check if variant passes quality filters
                if self._is_valid_variant(variant, problem, variants):
                    variants.append(variant)
                    
                    # Break if we have enough variants
                    if len(variants) >= count:
                        break
        
        return variants[:count]
    
    def _generate_variant_batch(
        self, 
        problem: str, 
        batch_size: int
    ) -> List[str]:
        """
        Generate a batch of variants using temperature cycling and persona-based prompting.
        
        Args:
            problem: The problem to generate variants for
            batch_size: Number of variants to generate
            
        Returns:
            List of generated variants
        """
        variants = []
        
        # Calculate variations needed per persona to achieve batch_size
        variants_per_persona = (batch_size + len(self.personas) - 1) // len(self.personas)
        
        for persona in self.personas:
            # Skip if we have enough variants
            if len(variants) >= batch_size:
                break
            
            # Select random transformation templates
            transformations = random.sample(
                self.transformations, 
                min(variants_per_persona, len(self.transformations))
            )
            
            for transformation in transformations:
                # Skip if we have enough variants
                if len(variants) >= batch_size:
                    break
                
                # Generate a temperature from the cycling range
                temperature = random.uniform(
                    self.config.TEMPERATURE_RANGE[0],
                    self.config.TEMPERATURE_RANGE[1]
                )
                
                # Build prompt with persona and transformation
                prompt = self._build_variant_prompt(
                    problem=problem,
                    persona=persona,
                    transformation=transformation
                )
                
                # Generate variant
                variant = self._generate_with_llm(prompt, temperature)
                
                if variant:
                    variants.append(variant)
        
        return variants
    
    def _build_variant_prompt(
        self, 
        problem: str, 
        persona: Dict[str, str], 
        transformation: str
    ) -> str:
        """
        Build a prompt for variant generation using a specific persona and transformation.
        
        Args:
            problem: The original problem
            persona: The mathematical persona to use
            transformation: The transformation template to apply
            
        Returns:
            Formatted prompt for the language model
        """
        # Extract properties from the problem for template filling
        problem_properties = self._extract_problem_properties(problem)
        
        # Fill transformation template with problem properties
        filled_transformation = transformation.format(**problem_properties)
        
        # Build the full prompt
        prompt = f"""
        {persona['description']}
        
        I need to create a simpler variant of the following calculus problem:
        
        ORIGINAL PROBLEM:
        {problem}
        
        HINT FOR SIMPLIFICATION:
        {filled_transformation}
        
        Generate a simplified version of this problem that is easier to solve while maintaining its mathematical essence.
        
        SIMPLIFIED PROBLEM:
        """
        
        return prompt.strip()
    
    def _extract_problem_properties(self, problem: str) -> Dict[str, str]:
        """
        Extract properties from a problem for template filling.
        
        This is a simplified version - in a full implementation, 
        it would use regex or more sophisticated parsing.
        
        Args:
            problem: The problem text
            
        Returns:
            Dictionary of properties extracted from the problem
        """
        # Simple extraction heuristics - would be more sophisticated in production
        properties = {
            "integrand": problem,
            "bounds": "[-∞, ∞]",
            "variable": "x",
            "suggestion": "x^2"
        }
        
        # Look for common patterns
        if "∫" in problem:
            # Try to extract the integrand between ∫ and dx
            parts = problem.split("∫")[1].split("dx")[0].strip()
            if parts:
                properties["integrand"] = parts
        
        if "[" in problem and "]" in problem:
            # Try to extract bounds
            bounds = problem.split("[")[1].split("]")[0].strip()
            if bounds:
                properties["bounds"] = f"[{bounds}]"
        
        return properties
    
    def _generate_with_llm(self, prompt: str, temperature: float) -> str:
        """
        Generate a completion using the language model.
        
        Args:
            prompt: The input prompt
            temperature: Temperature for generation
            
        Returns:
            Generated text
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=Config.ModelConfig.MAX_PROBLEM_LENGTH,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    num_return_sequences=1,
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            # Extract the generated problem from the output
            return self._extract_problem_from_output(generated_text)
            
        except Exception as e:
            logger.error(f"Error generating variant: {e}")
            return ""
    
    def _extract_problem_from_output(self, output: str) -> str:
        """
        Extract the problem statement from the generated output.
        
        Args:
            output: The raw generated text
            
        Returns:
            Extracted problem statement
        """
        # Strip any prefixes or suffixes to isolate the problem
        lines = output.strip().split("\n")
        
        # Simple extraction - would be more sophisticated in production
        filtered_lines = []
        for line in lines:
            # Skip common prefixes and empty lines
            if (line.strip().startswith("SIMPLIFIED PROBLEM") or 
                line.strip().startswith("Here's") or
                line.strip().startswith("Consider") or
                not line.strip()):
                continue
            filtered_lines.append(line)
        
        return "\n".join(filtered_lines).strip()
    
    def _is_valid_variant(
        self, 
        variant: str, 
        original_problem: str, 
        existing_variants: List[str]
    ) -> bool:
        """
        Check if a variant is valid and diverse enough.
        
        Args:
            variant: The variant to check
            original_problem: The original problem
            existing_variants: List of already generated variants
            
        Returns:
            True if the variant is valid, False otherwise
        """
        # Check if the variant is empty or too short
        if not variant or len(variant) < 10:
            return False
        
        # Check if the variant is too similar to the original problem
        if self._similarity(variant, original_problem) > self.config.SIMILARITY_THRESHOLD:
            return False
        
        # Check if the variant is too similar to existing variants
        for existing in existing_variants:
            if self._similarity(variant, existing) > self.config.SIMILARITY_THRESHOLD:
                return False
        
        # Additional validity checks would go here
        
        return True
    
    def _similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity between two strings.
        
        This is a very simple implementation - in production, you'd use
        embedding-based similarity or more sophisticated methods.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple character-level Jaccard similarity
        set1 = set(text1.lower())
        set2 = set(text2.lower())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0
