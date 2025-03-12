"""
Configuration file for the LADDER framework.
Contains parameters for all subsystems: variant generation, verification, RL, and TTRL.
"""

class Config:
    """Base configuration class for LADDER framework."""
    
    # General configurations
    PROJECT_NAME = "LADDER"
    DESCRIPTION = "Learning through Autonomous Difficulty-Driven Example Recursion"
    
    # Model configurations
    class ModelConfig:
        BASE_MODEL_NAME = "unsloth/Llama-3.2-3B"  # Initial implementation model
        PRODUCTION_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Production model
        MAX_PROBLEM_LENGTH = 256  # Maximum token length for problems
        MAX_SOLUTION_LENGTH = 512  # Maximum token length for solutions
        MIN_CONTEXT_WINDOW = 1024  # Minimum required context window
        
        # Deployment configurations
        TRAINING_QUANTIZATION = "4bit"  # 4-bit quantization with QLoRA for training
        INFERENCE_QUANTIZATION = "8bit"  # 8-bit quantization with flash attention for inference
    
    # Variant Generation Subsystem configurations
    class VGSConfig:
        # Structural properties
        MAX_DEPTH = 3  # Maximum depth of variant tree
        BRANCHING_FACTOR = 5  # Number of children per node
        
        # Generation parameters
        TEMPERATURE_RANGE = [0.8, 1.4]  # Temperature cycling range
        BATCH_SIZE = 10  # Variants generated per prompt
        DIFFICULTY_DISTRIBUTION = {
            "easier": 0.7,  # 70% easier variants
            "equivalent": 0.3  # 30% equivalent variants
        }
        
        # Diversity enhancement
        NUM_TRANSFORMATION_METHODS = 25  # Mathematical transformation methods
        NUM_PERSONAS = 5  # Number of mathematical perspectives for prompting
        
        # Quality filters
        SIMILARITY_THRESHOLD = 0.92  # Threshold for duplicate detection
    
    # Verification Subsystem configurations
    class VSConfig:
        # Numerical integration properties
        NUM_EVAL_POINTS = 5  # Number of points for evaluation
        EVAL_DOMAIN = [-10, 10]  # Domain for evaluation
        INTERVAL_WIDTH = 0.1  # Width of intervals for local evaluation
        RELATIVE_TOLERANCE = 1e-2  # Relative tolerance threshold
        
        # Error handling
        MAX_RETRY_COUNT = 3  # Maximum number of retry attempts
        TIMEOUT_THRESHOLD = 2  # Timeout threshold in seconds
    
    # Reinforcement Learning Subsystem configurations
    class RLSConfig:
        # GRPO configuration
        KL_COEFFICIENT = 0.001  # KL divergence coefficient
        CLIPPING_PARAMETER = 0.2  # Clipping parameter
        GROUP_SIZE = 8  # Outputs per problem for group normalization
        
        # Reward structure
        CORRECTNESS_REWARD = 1.0  # Reward for correct solution
        FORMAT_REWARD = 0.2  # Reward for proper formatting
        
        # Training hyperparameters
        BATCH_SIZE_3B = 32  # Batch size for 3B model
        BATCH_SIZE_7B = 128  # Batch size for 7B model
        LEARNING_RATE = 1e-5  # Learning rate
        MAX_TRAINING_STEPS = 5000  # Maximum training steps
    
    # Test-Time Reinforcement Learning Subsystem configurations
    class TTRLSConfig:
        # Variant generation for test problems
        MAX_DEPTH = 2  # Two-level tree for test problems
        MAX_VARIANTS = 800  # Maximum variants per test problem
        GENERATION_TIME_BUDGET = 60  # Time budget in seconds
        
        # RL properties
        MAX_RL_STEPS = 100  # Maximum RL steps per problem
        MICRO_BATCH_SIZE = 8  # Micro-batch size
        MAX_SOLUTION_ATTEMPTS = 10  # Maximum solution attempts after adaptation
        
        # Parameter management
        RESET_PARAMETERS_BETWEEN_PROBLEMS = True  # Reset parameters between test problems
        ENABLE_PARAMETER_CACHING = True  # Enable parameter caching for similar problems
    
    # Benchmark dataset configurations
    class BenchmarkConfig:
        DATASETS = {
            "integration_base": {
                "name": "LADDER Integration Base",
                "size": 110,
                "difficulty": "Undergraduate"
            },
            "mit_integration_bee": {
                "name": "MIT Integration Bee 2025",
                "size": 20,
                "difficulty": "Advanced undergraduate"
            },
            "calculus_standardized": {
                "name": "Calculus Standardized Tests",
                "size": 200,
                "difficulty": "High school to early undergraduate"
            },
            "custom_hard": {
                "name": "Custom Hard Integrals",
                "size": 50,
                "difficulty": "Graduate level"
            }
        }
        
        # Evaluation protocol
        TRAIN_VAL_TEST_SPLIT = [0.1, 0.1, 0.8]  # Split ratio
        CHECKPOINT_INTERVAL = 250  # Checkpoint every N steps
        EARLY_STOPPING_PATIENCE = 5  # Early stopping patience in checkpoints
