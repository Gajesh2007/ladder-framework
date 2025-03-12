"""
LADDER Framework - Main Driver Script

LADDER: Learning through Autonomous Difficulty-Driven Example Recursion

This script integrates all LADDER subsystems:
1. Variant Generation Subsystem (VGS)
2. Verification Subsystem (VS)
3. Reinforcement Learning Subsystem (RLS)
4. Test-Time Reinforcement Learning Subsystem (TTRL)

The framework enables self-improving language models through recursive problem decomposition.
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Any, Optional

import torch

from variant_generator.generator import VariantGenerator
from verification.verifier import IntegrationVerifier
from rl_subsystem.trainer import RLTrainer
from ttrl_subsystem.adaptor import TTRLAdaptor
from config.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ladder_framework.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="LADDER Framework for Self-Improving Language Models")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate variants
    variant_parser = subparsers.add_parser("generate-variants", help="Generate problem variants")
    variant_parser.add_argument("--problem", type=str, required=True, help="Root problem to generate variants for")
    variant_parser.add_argument("--output", type=str, default="variants.json", help="Output JSON file for variants")
    variant_parser.add_argument("--max-depth", type=int, default=3, help="Maximum depth of the variant tree")
    variant_parser.add_argument("--branching-factor", type=int, default=3, help="Branching factor for the variant tree")
    
    # Verify solutions
    verify_parser = subparsers.add_parser("verify", help="Verify integration solutions")
    verify_parser.add_argument("--problem", type=str, required=True, help="Integration problem to verify")
    verify_parser.add_argument("--solution", type=str, required=True, help="Proposed solution to verify")
    verify_parser.add_argument("--detailed", action="store_true", help="Provide detailed verification output")
    
    # Train model
    train_parser = subparsers.add_parser("train", help="Train model using GRPO")
    train_parser.add_argument("--dataset", type=str, required=True, help="Path to dataset of variants")
    train_parser.add_argument("--model", type=str, default=Config.ModelConfig.BASE_MODEL_NAME, help="Base model to train")
    train_parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory for trained model")
    train_parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for")
    train_parser.add_argument("--max-steps", type=int, default=1000, help="Maximum number of training steps")
    
    # Solve with TTRL adaptation
    ttrl_parser = subparsers.add_parser("solve", help="Solve a problem with TTRL adaptation")
    ttrl_parser.add_argument("--problem", type=str, required=True, help="Problem to solve")
    ttrl_parser.add_argument("--model", type=str, default=Config.ModelConfig.BASE_MODEL_NAME, help="Model to use for solving")
    ttrl_parser.add_argument("--max-variants", type=int, default=10, help="Maximum number of variants to generate")
    ttrl_parser.add_argument("--max-rl-steps", type=int, default=20, help="Maximum number of RL steps")
    
    # Generate full dataset
    dataset_parser = subparsers.add_parser("generate-dataset", help="Generate a full dataset of problems and variants")
    dataset_parser.add_argument("--input", type=str, required=True, help="Input file with root problems")
    dataset_parser.add_argument("--output", type=str, default="dataset.json", help="Output dataset file")
    dataset_parser.add_argument("--max-depth", type=int, default=3, help="Maximum depth of variant trees")
    dataset_parser.add_argument("--problems-per-tree", type=int, default=10, help="Maximum problems to sample per tree")
    
    # Run end-to-end pipeline
    pipeline_parser = subparsers.add_parser("pipeline", help="Run end-to-end LADDER pipeline")
    pipeline_parser.add_argument("--input", type=str, required=True, help="Input file with root problems")
    pipeline_parser.add_argument("--model", type=str, default=Config.ModelConfig.BASE_MODEL_NAME, help="Base model to train")
    pipeline_parser.add_argument("--output-dir", type=str, default="./pipeline_outputs", help="Output directory")
    pipeline_parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for")
    pipeline_parser.add_argument("--eval-problems", type=str, help="Problems to evaluate on after training")
    
    return parser.parse_args()

def generate_variants(args):
    """Generate problem variants."""
    logger.info(f"Generating variants for problem: {args.problem}")
    
    # Initialize variant generator
    generator = VariantGenerator()
    
    # Generate variant tree
    variant_tree = generator.generate_variant_tree(
        root_problem=args.problem,
        max_depth=args.max_depth,
        branching_factor=args.branching_factor
    )
    
    # Save to JSON file
    with open(args.output, 'w') as f:
        json.dump(variant_tree, f, indent=2)
    
    logger.info(f"Variant tree saved to {args.output}")
    
    # Print tree statistics
    count = count_variants_in_tree(variant_tree)
    logger.info(f"Generated {count} total variants in tree")
    
    return variant_tree

def count_variants_in_tree(tree):
    """Count total variants in a tree."""
    count = 1  # Count the root
    for child in tree.get("children", []):
        count += count_variants_in_tree(child)
    return count

def verify_solution(args):
    """Verify an integration solution."""
    logger.info(f"Verifying solution for problem: {args.problem}")
    
    # Initialize verifier
    verifier = IntegrationVerifier()
    
    # Verify solution
    result = verifier.verify_solution(
        problem=args.problem,
        solution=args.solution,
        detailed=args.detailed
    )
    
    # Print result
    if isinstance(result, bool):
        logger.info(f"Verification result: {'Correct' if result else 'Incorrect'}")
    else:
        logger.info(f"Verification result: {'Correct' if result['correct'] else 'Incorrect'}")
        if args.detailed:
            logger.info(f"Details: {json.dumps(result, indent=2)}")
    
    return result

def train_model(args):
    """Train model using GRPO."""
    logger.info(f"Training model {args.model} on dataset {args.dataset}")
    
    # Load dataset
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)
    
    # Prepare dataset for training
    train_dataset = []
    if isinstance(dataset, dict) and "problem" in dataset:
        # This is a variant tree
        flatten_variant_tree(dataset, train_dataset)
    else:
        # This is already a flat list
        train_dataset = dataset
    
    logger.info(f"Training on {len(train_dataset)} problems from dataset")
    
    # Initialize RL trainer
    trainer = RLTrainer(
        model_path=args.model,
        output_dir=args.output_dir
    )
    
    # Train model
    train_metrics = trainer.train(
        variant_dataset=train_dataset,
        num_epochs=args.epochs,
        max_steps=args.max_steps
    )
    
    # Log training results
    logger.info(f"Training complete. Final success rate: {train_metrics['success_rate'][-1]:.4f}")
    
    return train_metrics

def flatten_variant_tree(node, result):
    """Flatten a variant tree into a list of problems."""
    result.append({"problem": node["problem"]})
    for child in node.get("children", []):
        flatten_variant_tree(child, result)

def solve_with_ttrl(args):
    """Solve a problem with TTRL adaptation."""
    logger.info(f"Solving problem with TTRL adaptation: {args.problem}")
    
    # Initialize TTRL adaptor
    adaptor = TTRLAdaptor(model_path=args.model)
    
    # Solve with adaptation
    result = adaptor.solve_with_adaptation(
        problem=args.problem,
        max_variants=args.max_variants,
        max_rl_steps=args.max_rl_steps,
        verbose=True
    )
    
    # Log result
    if result["correct"]:
        logger.info("Problem solved correctly!")
    else:
        logger.info("Problem not solved correctly.")
    
    logger.info(f"Adaptation statistics: {json.dumps({k: v for k, v in result.items() if k != 'solution'}, indent=2)}")
    logger.info(f"Solution: {result['solution']}")
    
    return result

def generate_dataset(args):
    """Generate a full dataset of problems and variants."""
    logger.info(f"Generating dataset from root problems in {args.input}")
    
    # Load root problems
    with open(args.input, 'r') as f:
        root_problems = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(root_problems)} root problems")
    
    # Initialize variant generator
    generator = VariantGenerator()
    
    # Generate dataset
    dataset = []
    
    for i, problem in enumerate(root_problems, 1):
        logger.info(f"Generating variants for problem {i}/{len(root_problems)}: {problem}")
        
        # Generate variant tree
        try:
            variant_tree = generator.generate_variant_tree(
                root_problem=problem,
                max_depth=args.max_depth,
                branching_factor=max(2, args.problems_per_tree // args.max_depth)
            )
            
            # Sample problems from tree
            sampled_problems = []
            sample_from_tree(variant_tree, sampled_problems, args.problems_per_tree)
            
            # Add to dataset
            dataset.extend(sampled_problems)
            
            logger.info(f"Added {len(sampled_problems)} problems from tree {i}")
        except Exception as e:
            logger.error(f"Error generating variants for problem {problem}: {e}")
    
    # Shuffle dataset
    import random
    random.shuffle(dataset)
    
    # Save dataset
    with open(args.output, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    logger.info(f"Generated dataset with {len(dataset)} total problems, saved to {args.output}")
    
    return dataset

def sample_from_tree(node, result, max_samples):
    """Sample problems from a variant tree."""
    if len(result) >= max_samples:
        return
    
    # Add current problem
    result.append({"problem": node["problem"]})
    
    # Recursively sample from children
    children = node.get("children", [])
    for child in children:
        sample_from_tree(child, result, max_samples)

def run_pipeline(args):
    """Run end-to-end LADDER pipeline."""
    logger.info("Starting end-to-end LADDER pipeline")
    
    # 1. Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Generate dataset
    dataset_path = os.path.join(args.output_dir, "dataset.json")
    dataset_args = argparse.Namespace(
        input=args.input,
        output=dataset_path,
        max_depth=3,
        problems_per_tree=20
    )
    generate_dataset(dataset_args)
    
    # 3. Train model
    model_output_dir = os.path.join(args.output_dir, "trained_model")
    train_args = argparse.Namespace(
        dataset=dataset_path,
        model=args.model,
        output_dir=model_output_dir,
        epochs=args.epochs,
        max_steps=2000
    )
    train_model(train_args)
    
    # 4. Evaluate on test problems if provided
    if args.eval_problems:
        with open(args.eval_problems, 'r') as f:
            test_problems = [line.strip() for line in f if line.strip()]
        
        # Initialize TTRL adaptor with trained model
        adaptor = TTRLAdaptor(
            model_path=os.path.join(model_output_dir, "final_model")
        )
        
        # Evaluate each problem
        eval_results = []
        for problem in test_problems:
            result = adaptor.solve_with_adaptation(problem=problem)
            eval_results.append({
                "problem": problem,
                "result": result
            })
        
        # Calculate success rate
        success_rate = sum(1 for r in eval_results if r["result"]["correct"]) / len(eval_results)
        
        # Save evaluation results
        eval_path = os.path.join(args.output_dir, "evaluation_results.json")
        with open(eval_path, 'w') as f:
            json.dump({
                "success_rate": success_rate,
                "results": eval_results
            }, f, indent=2)
        
        logger.info(f"Evaluation complete. Success rate: {success_rate:.4f}")
    
    logger.info("Pipeline completed successfully")

def main():
    """Main entry point."""
    args = parse_args()
    
    # Choose command to run
    if args.command == "generate-variants":
        generate_variants(args)
    elif args.command == "verify":
        verify_solution(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "solve":
        solve_with_ttrl(args)
    elif args.command == "generate-dataset":
        generate_dataset(args)
    elif args.command == "pipeline":
        run_pipeline(args)
    else:
        logger.error("Please specify a command.")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        logger.exception("An error occurred:")
        exit(1)
