"""
Test-Time Reinforcement Learning Subsystem (TTRLS)

Enables on-the-fly adaptation to specific test problems through focused variant training.
Applies the same LADDER principles at inference time for improved performance.
"""

import os
import time
import copy
import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from variant_generator.generator import VariantGenerator
from verification.verifier import IntegrationVerifier
from rl_subsystem.trainer import RLTrainer
from config.config import Config

logger = logging.getLogger(__name__)

class TTRLAdaptor:
    """
    Test-Time Reinforcement Learning Adaptor for on-the-fly model adaptation.
    
    Key Features:
    - Dynamic variant generation for test problems
    - Micro-batch reinforcement learning for quick adaptation
    - Parameter reset between test problems to maintain a stable baseline
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_cache_dir: str = "./model_cache",
    ):
        """
        Initialize the Test-Time RL Adaptor.
        
        Args:
            model_path: Path to the pretrained LADDER-trained model
            tokenizer_path: Path to the tokenizer (defaults to model_path if None)
            device: Device to run the model on ("cuda" or "cpu")
            model_cache_dir: Directory to cache model parameters for similar problems
        """
        self.config = Config.TTRLSConfig
        self.device = device
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path if tokenizer_path else model_path
        self.model_cache_dir = model_cache_dir
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(model_cache_dir):
            os.makedirs(model_cache_dir)
        
        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        try:
            # First try with 8-bit quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype=torch.float16,
                load_in_8bit=True,  # Load model in 8-bit format for bitsandbytes compatibility
            )
        except (ImportError, RuntimeError) as e:
            logger.warning(f"Failed to load model with 8-bit quantization: {e}")
            logger.warning("Falling back to loading model without quantization...")
            # Fall back to no quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype=torch.float16,
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize variant generator
        self.variant_generator = VariantGenerator(
            model_path=self.model_path,
            tokenizer_path=self.tokenizer_path,
            device=self.device,
        )
        
        # Initialize verifier
        self.verifier = IntegrationVerifier()
        
        # Store original model parameters for resetting
        self.original_state_dict = copy.deepcopy(self.model.state_dict())
        
        # Initialize parameter cache
        self.parameter_cache = {}
        
        logger.info("TTRL Adaptor initialized successfully")
    
    def solve_with_adaptation(
        self,
        problem: str,
        max_variants: int = None,
        max_rl_steps: int = None,
        max_solution_attempts: int = None,
        time_budget: int = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Solve a problem with test-time adaptation through TTRL.
        
        Args:
            problem: The problem to solve
            max_variants: Maximum number of variants to generate
            max_rl_steps: Maximum number of RL steps to perform
            max_solution_attempts: Maximum number of solution attempts after adaptation
            time_budget: Time budget in seconds for the entire adaptation process
            verbose: Whether to print progress
            
        Returns:
            Dictionary with solution, correctness, and adaptation statistics
        """
        # Set default values from config if not provided
        if max_variants is None:
            max_variants = self.config.MAX_VARIANTS
        
        if max_rl_steps is None:
            max_rl_steps = self.config.MAX_RL_STEPS
        
        if max_solution_attempts is None:
            max_solution_attempts = self.config.MAX_SOLUTION_ATTEMPTS
        
        if time_budget is None:
            time_budget = self.config.GENERATION_TIME_BUDGET
        
        start_time = time.time()
        
        # Check if we have a cached model for a similar problem
        cache_hit, cached_state_dict = self._check_parameter_cache(problem)
        if cache_hit:
            logger.info("Using cached model parameters for similar problem")
            self.model.load_state_dict(cached_state_dict)
        
        # Try solving directly first
        initial_solution, initial_correct = self._generate_and_verify_solution(problem)
        
        if initial_correct:
            logger.info("Problem solved correctly without adaptation")
            return {
                "solution": initial_solution,
                "correct": True,
                "adaptation_performed": False,
                "variants_generated": 0,
                "rl_steps": 0,
                "time_taken": time.time() - start_time,
            }
        
        # Generate variants if direct solution failed
        logger.info("Initial solution incorrect, generating variants for adaptation")
        variants = self._generate_problem_variants(
            problem, max_variants, max_depth=self.config.MAX_DEPTH
        )
        
        num_variants = len(variants)
        if verbose:
            logger.info(f"Generated {num_variants} variants for TTRL")
        
        # Skip adaptation if no variants were generated
        if num_variants == 0:
            logger.warning("No variants generated, skipping adaptation")
            return {
                "solution": initial_solution,
                "correct": False,
                "adaptation_performed": False,
                "variants_generated": 0,
                "rl_steps": 0,
                "time_taken": time.time() - start_time,
            }
        
        # Perform RL on variants
        rl_steps = self._perform_ttrl(variants, max_rl_steps, verbose)
        
        # Try solving again with adapted model
        best_solution = None
        best_correct = False
        
        for attempt in range(max_solution_attempts):
            solution, correct = self._generate_and_verify_solution(problem)
            
            if correct:
                best_solution = solution
                best_correct = True
                break
            
            if best_solution is None or len(solution) > len(best_solution):
                best_solution = solution
        
        # Record adaptation time
        adaptation_time = time.time() - start_time
        
        # Cache model parameters if adaptation was successful
        if best_correct and self.config.ENABLE_PARAMETER_CACHING:
            self._cache_parameters(problem)
        
        # Reset model parameters if configured to do so
        if self.config.RESET_PARAMETERS_BETWEEN_PROBLEMS:
            self._reset_model_parameters()
        
        return {
            "solution": best_solution if best_solution is not None else initial_solution,
            "correct": best_correct,
            "adaptation_performed": True,
            "variants_generated": num_variants,
            "rl_steps": rl_steps,
            "time_taken": adaptation_time,
        }
    
    def _generate_problem_variants(
        self,
        problem: str,
        max_variants: int,
        max_depth: int = 2,
    ) -> List[Dict[str, str]]:
        """
        Generate variants for a test problem.
        
        Args:
            problem: The problem to generate variants for
            max_variants: Maximum number of variants to generate
            max_depth: Maximum depth of the variant tree
            
        Returns:
            List of variant problems
        """
        # Generate variant tree
        variant_tree = self.variant_generator.generate_variant_tree(
            root_problem=problem,
            max_depth=max_depth,
            branching_factor=min(5, max(1, max_variants // (max_depth * 2))),
        )
        
        # Flatten tree into a list of variants
        variants = []
        self._flatten_variant_tree(variant_tree, variants)
        
        # Limit number of variants
        if len(variants) > max_variants:
            variants = variants[:max_variants]
        
        return variants
    
    def _flatten_variant_tree(
        self,
        node: Dict[str, Any],
        result: List[Dict[str, str]],
    ) -> None:
        """
        Flatten a variant tree into a list of variants.
        
        Args:
            node: Current node in the variant tree
            result: List to store flattened variants
        """
        # Add current node to result
        result.append({"problem": node["problem"]})
        
        # Process children
        for child in node.get("children", []):
            self._flatten_variant_tree(child, result)
    
    def _perform_ttrl(
        self,
        variants: List[Dict[str, str]],
        max_rl_steps: int,
        verbose: bool = True,
    ) -> int:
        """
        Perform Test-Time Reinforcement Learning on a set of variants.
        
        Args:
            variants: List of variant problems
            max_rl_steps: Maximum number of RL steps to perform
            verbose: Whether to print progress
            
        Returns:
            Number of RL steps performed
        """
        # Create a temporary directory for TTRL outputs
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a lightweight RL trainer for TTRL
            ttrl_trainer = RLTrainer(
                model_path=self.model_path,  # Use path to avoid duplicating the model
                tokenizer_path=self.tokenizer_path,
                output_dir=temp_dir,
                device=self.device,
            )
            
            # Copy current model weights to trainer
            ttrl_trainer.model.load_state_dict(self.model.state_dict())
            
            # Perform RL training with micro-batches
            train_metrics = ttrl_trainer.train(
                variant_dataset=variants,
                num_epochs=1,
                max_steps=max_rl_steps,
                eval_interval=max_rl_steps + 1,  # Disable intermediate evaluation
                checkpoint_interval=max_rl_steps + 1,  # Disable intermediate checkpointing
                early_stopping_patience=max_rl_steps + 1,  # Disable early stopping
                verbose=verbose,
            )
            
            # Copy trained weights back to main model
            self.model.load_state_dict(ttrl_trainer.model.state_dict())
            
            # Return number of steps performed
            return min(max_rl_steps, len(train_metrics.get("step", [])))
    
    def _generate_and_verify_solution(
        self,
        problem: str,
    ) -> Tuple[str, bool]:
        """
        Generate and verify a solution for a problem.
        
        Args:
            problem: The problem to solve
            
        Returns:
            Tuple of (solution, correctness)
        """
        # Build prompt
        prompt = f"""Solve the following integral. Express the answer in its simplest form and place it within <ANSWER> tags.

PROBLEM:
{problem}

SOLUTION:
"""
        
        # Format inputs
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate solution
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=Config.ModelConfig.MAX_SOLUTION_LENGTH,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode solution
        solution = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # Verify solution
        correct = self.verifier.verify_solution(problem, solution)
        
        return solution, correct
    
    def _check_parameter_cache(
        self,
        problem: str,
    ) -> Tuple[bool, Optional[Dict[str, torch.Tensor]]]:
        """
        Check if we have cached parameters for a similar problem.
        
        Args:
            problem: The problem to check
            
        Returns:
            Tuple of (cache hit boolean, cached parameters if hit)
        """
        if not self.config.ENABLE_PARAMETER_CACHING:
            return False, None
        
        # Simple caching heuristic
        # In a full implementation, this would use problem embedding similarity
        for cached_problem, state_dict_path in self.parameter_cache.items():
            if self._problems_are_similar(problem, cached_problem):
                try:
                    state_dict = torch.load(state_dict_path, map_location=self.device)
                    return True, state_dict
                except Exception as e:
                    logger.error(f"Error loading cached parameters: {e}")
                    return False, None
        
        return False, None
    
    def _problems_are_similar(
        self,
        problem1: str,
        problem2: str,
    ) -> bool:
        """
        Check if two problems are similar.
        
        Args:
            problem1: First problem
            problem2: Second problem
            
        Returns:
            Boolean indicating similarity
        """
        # Simple similarity heuristic
        # In a full implementation, this would use problem embedding similarity
        import re
        
        # Extract mathematical expressions
        def extract_math(text):
            # Extract expressions between $, \(, \), etc.
            math_expressions = re.findall(r'\$(.*?)\$|\\\((.*?)\\\)|\\\[(.*?)\\\]', text)
            # Flatten the list of tuples and filter out empty strings
            return [expr for tuple_expr in math_expressions for expr in tuple_expr if expr]
        
        math1 = extract_math(problem1)
        math2 = extract_math(problem2)
        
        # Check if they share at least one expression
        return bool(set(math1).intersection(set(math2)))
    
    def _cache_parameters(
        self,
        problem: str,
    ) -> None:
        """
        Cache current model parameters for a problem.
        
        Args:
            problem: The problem to cache parameters for
        """
        if not self.config.ENABLE_PARAMETER_CACHING:
            return
        
        # Generate a unique identifier for the problem
        import hashlib
        problem_hash = hashlib.md5(problem.encode()).hexdigest()
        
        # Create cache file path
        cache_path = os.path.join(self.model_cache_dir, f"params_{problem_hash}.pt")
        
        # Save model parameters
        torch.save(self.model.state_dict(), cache_path)
        
        # Update cache dictionary
        self.parameter_cache[problem] = cache_path
        
        # Limit cache size
        if len(self.parameter_cache) > 20:  # Arbitrary limit
            # Remove oldest entry
            oldest_problem = next(iter(self.parameter_cache))
            oldest_path = self.parameter_cache.pop(oldest_problem)
            
            # Delete file if it exists
            if os.path.exists(oldest_path):
                os.remove(oldest_path)
    
    def _reset_model_parameters(self) -> None:
        """
        Reset model parameters to the original state.
        """
        if self.config.RESET_PARAMETERS_BETWEEN_PROBLEMS:
            self.model.load_state_dict(self.original_state_dict)
            logger.info("Model parameters reset to original state")
    
    def _load_domain_model(self, new_model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load a domain-specific model for a particular class of problems.
        
        Args:
            new_model_path: Path to the domain-specific model
            
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading domain model from {new_model_path}")
        
        try:
            # First try with 8-bit quantization
            new_model = AutoModelForCausalLM.from_pretrained(
                new_model_path,
                device_map=self.device,
                torch_dtype=torch.float16,
                load_in_8bit=True,
            )
        except (ImportError, RuntimeError) as e:
            logger.warning(f"Failed to load domain model with 8-bit quantization: {e}")
            logger.warning("Falling back to loading domain model without quantization...")
            # Fall back to no quantization
            new_model = AutoModelForCausalLM.from_pretrained(
                new_model_path,
                device_map=self.device,
                torch_dtype=torch.float16,
            )
        
        new_tokenizer = AutoTokenizer.from_pretrained(
            new_model_path,
            padding_side="left",
        )
        new_tokenizer.pad_token = new_tokenizer.eos_token
        
        return new_model, new_tokenizer
    
    def update_base_model(
        self,
        new_model_path: str,
        new_tokenizer_path: Optional[str] = None,
    ) -> None:
        """
        Update the base model used by the TTRL adaptor.
        
        Args:
            new_model_path: Path to the new model
            new_tokenizer_path: Path to the new tokenizer (defaults to new_model_path if None)
        """
        logger.info(f"Updating base model to {new_model_path}")
        
        # Load new model and tokenizer
        new_model, new_tokenizer = self._load_domain_model(new_model_path)
        
        # Update model and tokenizer
        self.model = new_model
        self.tokenizer = new_tokenizer
        
        # Update model path
        self.model_path = new_model_path
        self.tokenizer_path = new_tokenizer_path if new_tokenizer_path else new_model_path
        
        # Store original state dict for resetting
        self.original_state_dict = copy.deepcopy(self.model.state_dict())
        
        # Update variant generator
        self.variant_generator = VariantGenerator(
            model_path=self.model_path,
            tokenizer_path=self.tokenizer_path,
            device=self.device,
        )
        
        # Clear parameter cache
        self.parameter_cache = {}
        
        logger.info("Base model updated successfully")
