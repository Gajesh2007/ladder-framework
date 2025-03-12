"""
Reinforcement Learning Subsystem (RLS)

Implements Group Relative Policy Optimization (GRPO) on generated variants to train the base model.
Processes variant trees to create a natural difficulty gradient for effective RL training.
"""

import os
import time
import math
import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import PPOTrainer, PPOConfig

from verification.verifier import IntegrationVerifier
from config.config import Config

logger = logging.getLogger(__name__)

class RLTrainer:
    """
    Reinforcement Learning Trainer using Group Relative Policy Optimization (GRPO).
    
    Key Features:
    - Implementation of GRPO with KL regularization to the reference model
    - Group-based advantage normalization
    - Support for data loading from variant trees
    - Checkpointing and evaluation
    """
    
    def __init__(
        self,
        model_path: str = Config.ModelConfig.BASE_MODEL_NAME,
        tokenizer_path: Optional[str] = None,
        output_dir: str = "./outputs",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        device_map: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
    ):
        """
        Initialize the RL Trainer with a language model and configuration.
        
        Args:
            model_path: Path to the pretrained language model
            tokenizer_path: Path to the tokenizer (defaults to model_path if None)
            output_dir: Directory to save checkpoints and outputs
            device: Device to run the model on ("cuda" or "cpu")
            device_map: Device mapping strategy for distributed training
            load_in_8bit: Whether to load the model in 8-bit precision
            load_in_4bit: Whether to load the model in 4-bit precision
        """
        self.config = Config.RLSConfig
        self.device = device
        self.output_dir = output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        
        # Try to load the model with the requested quantization, with fallbacks
        try:
            # Determine quantization approach
            quantization_config = None
            if load_in_4bit:
                quantization_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            elif load_in_8bit:
                quantization_config = transformers.BitsAndBytesConfig(
                    load_in_8bit=True
                )
            
            # Load the model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
            )
        except (ImportError, RuntimeError) as e:
            logger.warning(f"Failed to load model with quantization: {e}")
            logger.warning("Falling back to loading model without quantization...")
            # Try 8-bit if 4-bit failed
            if load_in_4bit and not load_in_8bit:
                try:
                    logger.info("Attempting to load with 8-bit quantization instead")
                    quantization_config = transformers.BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map=device_map,
                        quantization_config=quantization_config,
                        torch_dtype=torch.float16,
                    )
                except (ImportError, RuntimeError) as e:
                    logger.warning(f"8-bit quantization also failed: {e}")
                    # Fall back to no quantization
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map=device_map,
                        torch_dtype=torch.float16,
                    )
            else:
                # Fall back to no quantization
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device_map,
                    torch_dtype=torch.float16,
                )
        
        # Prepare model for k-bit training if quantization was successful
        try:
            if (load_in_4bit or load_in_8bit) and quantization_config is not None:
                self.model = prepare_model_for_kbit_training(self.model)
        except Exception as e:
            logger.warning(f"Failed to prepare model for k-bit training: {e}")
            logger.warning("Continuing without k-bit training preparation...")
        
        # Add LoRA adapters for parameter-efficient fine-tuning
        try:
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            self.model = get_peft_model(self.model, peft_config)
        except Exception as e:
            logger.warning(f"Failed to add LoRA adapters: {e}")
            logger.warning("Continuing without LoRA adapters...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path else model_path,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize PPO configuration
        ppo_config = PPOConfig(
            learning_rate=self.config.LEARNING_RATE,
            batch_size=self.config.BATCH_SIZE_3B if "3B" in model_path else self.config.BATCH_SIZE_7B,
            mini_batch_size=8,
            gradient_accumulation_steps=1,
            ppo_epochs=4,
            max_grad_norm=1.0,
            seed=42,
            target_kl=self.config.KL_COEFFICIENT,
            use_score_scaling=True,
            use_score_norm=True,
            kl_penalty="kl",
            cliprange=self.config.CLIPPING_PARAMETER,
            cliprange_value=self.config.CLIPPING_PARAMETER,
            vf_coef=0.1,
            horizon=10000,
            gamma=1.0,
            lam=0.95,
            init_kl_coef=0.001,
            adap_kl_ctrl=True,
        )
        
        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        # Initialize verifier
        self.verifier = IntegrationVerifier()
        
        # Initialize training metrics
        self.metrics = {
            "train_loss": [],
            "policy_loss": [],
            "value_loss": [],
            "kl_div": [],
            "learning_rate": [],
            "reward_mean": [],
            "reward_std": [],
            "step": [],
            "epoch": [],
            "success_rate": [],
            "evaluation_metrics": {}
        }
        
        logger.info("RL trainer initialized successfully")
    
    def train(
        self,
        variant_dataset: Union[str, List[Dict[str, Any]]],
        num_epochs: int = 1,
        max_steps: int = None,
        eval_interval: int = 250,
        checkpoint_interval: int = 500,
        early_stopping_patience: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the model using GRPO on the variant dataset.
        
        Args:
            variant_dataset: Dataset of variants, either as a path to a JSON file or a list of dictionaries
            num_epochs: Number of epochs to train for
            max_steps: Maximum number of steps (if None, train for num_epochs)
            eval_interval: Interval for evaluation
            checkpoint_interval: Interval for checkpointing
            early_stopping_patience: Patience for early stopping in evaluation steps
            verbose: Whether to print progress
            
        Returns:
            Training metrics
        """
        # Load dataset
        if isinstance(variant_dataset, str):
            import json
            with open(variant_dataset, 'r') as f:
                dataset = json.load(f)
        else:
            dataset = variant_dataset
        
        logger.info(f"Training on dataset with {len(dataset)} samples")
        
        # Set max steps
        if max_steps is None:
            max_steps = self.config.MAX_TRAINING_STEPS
        
        # Initialize training state
        step = 0
        epoch = 0
        best_eval_score = 0.0
        patience_counter = 0
        
        # Training loop
        while epoch < num_epochs and step < max_steps:
            epoch_start_time = time.time()
            
            # Shuffle dataset
            np.random.shuffle(dataset)
            
            # Process dataset in batches
            batch_size = self.ppo_trainer.config.batch_size
            for batch_idx in range(0, len(dataset), batch_size):
                # Check if max steps reached
                if step >= max_steps:
                    break
                
                # Get current batch
                batch_end_idx = min(batch_idx + batch_size, len(dataset))
                current_batch = dataset[batch_idx:batch_end_idx]
                
                # Process batch for GRPO
                batch_metrics = self._process_batch_grpo(current_batch, step)
                
                # Update metrics
                for key, value in batch_metrics.items():
                    if key in self.metrics:
                        self.metrics[key].append(value)
                
                # Record step
                self.metrics["step"].append(step)
                
                # Print progress
                if verbose and step % 10 == 0:
                    self._print_progress(step, epoch, batch_metrics)
                
                # Evaluation
                if step % eval_interval == 0 and step > 0:
                    eval_metrics = self._evaluate()
                    self.metrics["evaluation_metrics"][step] = eval_metrics
                    
                    # Early stopping check
                    current_eval_score = eval_metrics["success_rate"]
                    if current_eval_score > best_eval_score:
                        best_eval_score = current_eval_score
                        patience_counter = 0
                        
                        # Save best model
                        self._save_checkpoint(step, is_best=True)
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping triggered after {step} steps")
                        break
                
                # Checkpointing
                if step % checkpoint_interval == 0 and step > 0:
                    self._save_checkpoint(step)
                
                step += 1
            
            # Record epoch
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")
            self.metrics["epoch"].append(epoch)
            
            epoch += 1
        
        # Final evaluation
        final_eval_metrics = self._evaluate()
        self.metrics["evaluation_metrics"][step] = final_eval_metrics
        
        # Save final model
        self._save_checkpoint(step, is_final=True)
        
        logger.info(f"Training completed after {step} steps")
        
        return self.metrics
    
    def _process_batch_grpo(
        self, 
        batch: List[Dict[str, Any]], 
        step: int
    ) -> Dict[str, float]:
        """
        Process a batch using Group Relative Policy Optimization (GRPO).
        
        Args:
            batch: List of samples, each with a problem and variants
            step: Current training step
            
        Returns:
            Batch metrics
        """
        # Initialize batch metrics
        batch_metrics = {
            "train_loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "kl_div": 0.0,
            "reward_mean": 0.0,
            "reward_std": 0.0,
            "success_rate": 0.0
        }
        
        # Group the batch for GRPO
        groups = []
        group_size = self.config.GROUP_SIZE
        
        # Create groups from the batch
        for i in range(0, len(batch), group_size):
            group_end = min(i + group_size, len(batch))
            group = batch[i:group_end]
            
            if len(group) == group_size:  # Only use full groups
                groups.append(group)
        
        total_correct = 0
        total_samples = 0
        
        # Process each group
        for group in groups:
            # Prepare group for PPO
            queries, responses, rewards = self._prepare_group(group)
            
            # Record success rate
            total_correct += sum(r > 0 for r in rewards)
            total_samples += len(rewards)
            
            # Process the group with PPO
            train_stats = self.ppo_trainer.step(queries, responses, rewards)
            
            # Update batch metrics
            batch_metrics["train_loss"] += train_stats["train/loss"]
            batch_metrics["policy_loss"] += train_stats["train/policy_loss"]
            batch_metrics["value_loss"] += train_stats["train/value_loss"]
            batch_metrics["kl_div"] += train_stats["train/kl"]
            batch_metrics["reward_mean"] += np.mean(rewards)
            batch_metrics["reward_std"] += np.std(rewards)
        
        # Average metrics across groups
        num_groups = len(groups)
        if num_groups > 0:
            for key in batch_metrics:
                if key != "success_rate":
                    batch_metrics[key] /= num_groups
        
        # Calculate success rate
        if total_samples > 0:
            batch_metrics["success_rate"] = total_correct / total_samples
        
        return batch_metrics
    
    def _prepare_group(
        self, 
        group: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str], List[float]]:
        """
        Prepare a group of samples for PPO training.
        
        Args:
            group: List of samples in a group
            
        Returns:
            Tuple of (queries, responses, rewards)
        """
        queries = []
        responses = []
        rewards = []
        
        # Process each sample in the group
        for sample in group:
            # Get problem and generate solution
            problem = sample["problem"]
            query = self._build_problem_prompt(problem)
            
            # Generate solution with the current model
            response = self._generate_solution(query)
            
            # Verify solution correctness
            is_correct = self.verifier.verify_solution(problem, response)
            
            # Calculate reward
            reward = self._calculate_reward(response, is_correct)
            
            # Add to lists
            queries.append(query)
            responses.append(response)
            rewards.append(reward)
        
        # Normalize rewards within the group (GRPO)
        normalized_rewards = self._normalize_rewards_grpo(rewards)
        
        return queries, responses, normalized_rewards
    
    def _build_problem_prompt(self, problem: str) -> str:
        """
        Build a prompt for a problem.
        
        Args:
            problem: The problem statement
            
        Returns:
            Formatted prompt
        """
        return f"""Solve the following integral. Express the answer in its simplest form and place it within <ANSWER> tags.

PROBLEM:
{problem}

SOLUTION:
"""
    
    def _generate_solution(self, prompt: str) -> str:
        """
        Generate a solution for a problem using the current model.
        
        Args:
            prompt: The problem prompt
            
        Returns:
            Generated solution
        """
        # Format inputs
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
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
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return response
    
    def _calculate_reward(self, response: str, is_correct: bool) -> float:
        """
        Calculate reward for a response.
        
        Args:
            response: The generated response
            is_correct: Whether the response is correct
            
        Returns:
            Reward value
        """
        # Base reward for correctness
        reward = self.config.CORRECTNESS_REWARD if is_correct else 0.0
        
        # Format reward for using answer tags
        if "<ANSWER>" in response and "</ANSWER>" in response:
            reward += self.config.FORMAT_REWARD
        
        return reward
    
    def _normalize_rewards_grpo(self, rewards: List[float]) -> List[float]:
        """
        Normalize rewards using Group Relative Policy Optimization (GRPO).
        
        Args:
            rewards: List of rewards
            
        Returns:
            Normalized rewards
        """
        # Calculate group statistics
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards) + 1e-8  # Add small epsilon for numerical stability
        
        # Normalize rewards
        normalized_rewards = [(r - reward_mean) / reward_std for r in rewards]
        
        return normalized_rewards
    
    def _evaluate(self) -> Dict[str, float]:
        """
        Evaluate the current model on a validation set.
        
        Returns:
            Evaluation metrics
        """
        # This would be more sophisticated in a full implementation,
        # loading a validation set and evaluating the model
        
        # Placeholder for now - just return the latest success rate
        if len(self.metrics["success_rate"]) > 0:
            return {"success_rate": self.metrics["success_rate"][-1]}
        else:
            return {"success_rate": 0.0}
    
    def _save_checkpoint(
        self, 
        step: int, 
        is_best: bool = False, 
        is_final: bool = False
    ) -> None:
        """
        Save a model checkpoint.
        
        Args:
            step: Current training step
            is_best: Whether this is the best model so far
            is_final: Whether this is the final model
        """
        # Determine checkpoint path
        if is_final:
            checkpoint_dir = os.path.join(self.output_dir, "final_model")
        elif is_best:
            checkpoint_dir = os.path.join(self.output_dir, "best_model")
        else:
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        
        # Create directory if it doesn't exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training metrics
        import json
        metrics_path = os.path.join(checkpoint_dir, "training_metrics.json")
        
        # Convert metrics to JSON-serializable format
        serializable_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                serializable_metrics[key] = [v.tolist() for v in value]
            else:
                serializable_metrics[key] = value
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def _print_progress(
        self, 
        step: int, 
        epoch: int, 
        metrics: Dict[str, float]
    ) -> None:
        """
        Print training progress.
        
        Args:
            step: Current training step
            epoch: Current epoch
            metrics: Batch metrics
        """
        logger.info(
            f"Step {step} | Epoch {epoch} | "
            f"Loss: {metrics['train_loss']:.4f} | "
            f"Policy Loss: {metrics['policy_loss']:.4f} | "
            f"Value Loss: {metrics['value_loss']:.4f} | "
            f"KL Div: {metrics['kl_div']:.4f} | "
            f"Reward Mean: {metrics['reward_mean']:.4f} | "
            f"Success Rate: {metrics['success_rate']:.4f}"
        )
