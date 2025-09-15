#!/usr/bin/env python3
"""
Rust Code Training with GRPO (Group Relative Policy Optimization)

This script trains a language model to generate Rust code using cargo toolchain 
feedback as rewards for reinforcement learning.
"""

import os
import re
import json
import time
import shutil
import subprocess
import functools
from pathlib import Path
from uuid import uuid4
from typing import Any, Callable, Optional
from datetime import datetime

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model


# System prompt for the model
SYSTEM_PROMPT = """You are a pragmatic Rust programmer who enjoys test driven development. Given the following question, write a Rust function to complete the task. Make the code simple and easy to understand. The code should pass `cargo build` and `cargo clippy`. Try to limit library usage to the standard library std. Be careful with your types, and try to limit yourself to the basic built in types and standard library functions. When writing the function you can think through how to solve the problem and perform reasoning in the comments above the function.

Then write unit tests for the function you defined. Write multiple unit tests for the function. The tests should be a simple line delimited list of assert! or assert_eq! statements. When writing the unit tests you can have comments specifying what you are testing in plain english. The tests should use super::*.

An example output should look like the following:

```rust
/// Reasoning goes here
/// and can be multi-line
fn add_nums(x: i32, y: i32) -> i32 {
  x + y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_nums() {
        // Test adding positive numbers
        assert_eq!(add_nums(4, 2), 6);
        // Test adding a positive and negative number
        assert_eq!(add_nums(4, -2), 2);
        // Test adding two negative numbers
        assert_eq!(add_nums(-12, -1), -13);
    }
}
```

Make sure to only respond with a single  ```rust``` block. The unit tests must be defined inside the mod tests {} module. Make sure to import any standard library modules that you need. Do not add a main function.
"""


class RustTool:
    """Tool for running Rust cargo commands (build, test, clippy)"""
    
    def __init__(self, name: str):
        self.name = name

    def run(self, results: dict, project_dir: Path) -> dict:
        """Run cargo command and update results dict"""
        try:
            result = subprocess.run(
                ["cargo", self.name, "--quiet"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            results[f'{self.name}_passed'] = result.returncode == 0
            results[f'{self.name}_stderr'] = str(result.stderr)
        except Exception as e:
            results[f'{self.name}_passed'] = False
            results[f'{self.name}_stderr'] = f"{e}"
        return results


class RustCodeEvaluator:
    """Evaluates Rust code using cargo toolchain and various heuristics"""
    
    @staticmethod
    def extract_regex(text: str, pattern: str) -> Optional[str]:
        """Extract text using regex pattern"""
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1) if match else None

    @staticmethod
    def extract_rust_code(response: str) -> str:
        """Extract Rust code from markdown code blocks"""
        pattern = r'```rust\n(.*?)\n```'
        code = RustCodeEvaluator.extract_regex(response, pattern)
        return code if code else response

    @staticmethod
    def extract_test_code(response: str) -> str:
        """Extract test module from Rust code"""
        pattern = r'(#\[cfg\(test\)\]\s*mod\s+tests\s*\{.*?\})'
        return RustCodeEvaluator.extract_regex(response, pattern)

    @staticmethod
    def get_cargo_template() -> str:
        """Get Cargo.toml template"""
        return """
[package]
name = "rust-program"
version = "0.1.0"
edition = "2021"

[dependencies]
"""

    @staticmethod
    def get_rust_template() -> str:
        """Get main.rs template"""
        return """
#![allow(dead_code)]
// {code}

// Need basic main function for the code to compile
fn main() {
  println!("Hello World");
}
"""

    def setup_and_test_rust_project(self, row: dict, tools: list) -> dict:
        """
        Set up temporary Rust project and run tests
        
        This is where the actual Rust code execution happens:
        1. Creates a temporary Rust project structure
        2. Injects the model-generated code into a main.rs file
        3. Runs cargo commands (build/test/clippy) on the real code
        4. Returns results to be used as RL rewards
        """
        # Create temporary project directory with unique name
        project_dir = Path("outputs") / "tests" / f"temp_rust_project_{uuid4()}"
        project_dir_src = project_dir / "src"
        project_dir_src.mkdir(parents=True, exist_ok=True)

        # Extract and prepare the model-generated Rust code
        template = self.get_rust_template()
        rust_code = self.extract_rust_code(row['rust_code'])  # Extract from ```rust``` blocks
        template = template.replace("// {code}", rust_code)

        # Write the actual Rust source file
        main_rs_path = project_dir_src / "main.rs"
        with open(main_rs_path, "w") as f:
            f.write(template)
            
        print(f"Created Rust project at: {project_dir}")
        print(f"Generated code:\n{template}")

        # Write Cargo.toml to make it a valid Rust project
        cargo_file_path = project_dir / "Cargo.toml"
        with open(cargo_file_path, "w") as f:
            f.write(self.get_cargo_template())

        # Actually run the cargo tools on the generated code
        results = {}
        for tool in tools:
            print(f"Running: cargo {tool.name}")
            results = tool.run(results, project_dir)
            print(f"Result: {results}")

        # Clean up the temporary project
        shutil.rmtree(project_dir)
        return results


class RewardFunctions:
    """Collection of reward functions for GRPO training"""
    
    def __init__(self, evaluator: RustCodeEvaluator):
        self.evaluator = evaluator

    def create_reward_function(self, name: str, func: Callable):
        """Create a logged reward function"""
        def reward_func(prompts, completions, **kwargs) -> list[float]:
            contents = [completion[0]["content"] for completion in completions]
            return func(contents, **kwargs)
        
        reward_func.__name__ = name
        return reward_func

    def non_empty_reward(self, contents: list[str], **kwargs) -> list[float]:
        """Reward for non-empty responses with proper structure"""
        results = []
        for content in contents:
            if not (self._has_code_block(content) and self._has_test_block(content)):
                results.append(0.0)
                continue
                
            code = self.evaluator.extract_rust_code(content)
            num_non_empty = 0
            for line in code.split("\n"):
                line = line.strip()
                if line.startswith("//") or len(line) < 2:
                    continue
                num_non_empty += 1
            results.append(1.0 if num_non_empty >= 3 else 0.0)
        return results

    def test_asserts_reward(self, contents: list[str], **kwargs) -> list[float]:
        """Reward for having assert statements in tests"""
        results = []
        for content in contents:
            test_code = self.evaluator.extract_test_code(content)
            if not test_code:
                results.append(0.0)
                continue

            unique_asserts = set()
            for line in test_code.split("\n"):
                line = line.strip()
                if line.startswith("assert!(") or line.startswith("assert_eq!("):
                    unique_asserts.add(line)
            
            if len(unique_asserts) >= 4:
                results.append(1.0)
            else:
                results.append(0.25 * len(unique_asserts))
        return results

    def code_block_reward(self, contents: list[str], **kwargs) -> list[float]:
        """Reward for having proper code blocks"""
        return [0.5 if self._has_code_block(content) else 0.0 for content in contents]

    def test_block_reward(self, contents: list[str], **kwargs) -> list[float]:
        """Reward for having test blocks"""
        return [0.5 if self._has_test_block(content) else 0.0 for content in contents]

    def cargo_build_reward(self, contents: list[str], **kwargs) -> list[float]:
        """Reward for passing cargo build"""
        results = []
        for content in contents:
            code = self.evaluator.extract_rust_code(content)
            data = {'rust_code': code}
            tools = [RustTool("build")]
            cargo_results = self.evaluator.setup_and_test_rust_project(data, tools)
            score = 1.0 if cargo_results['build_passed'] else 0.0
            results.append(score)
        return results

    def cargo_clippy_reward(self, contents: list[str], **kwargs) -> list[float]:
        """Reward for passing cargo clippy"""
        results = []
        for content in contents:
            code = self.evaluator.extract_rust_code(content)
            data = {'rust_code': code}
            tools = [RustTool("clippy")]
            cargo_results = self.evaluator.setup_and_test_rust_project(data, tools)
            score = 1.0 if cargo_results['clippy_passed'] else 0.0
            results.append(score)
        return results

    def cargo_test_reward(self, contents: list[str], **kwargs) -> list[float]:
        """Reward for passing cargo test (weighted higher)"""
        results = []
        for content in contents:
            code = self.evaluator.extract_rust_code(content)
            test_code = self.evaluator.extract_test_code(code)
            
            score = 0.0
            if test_code:
                data = {'rust_code': code}
                tools = [RustTool("test")]
                cargo_results = self.evaluator.setup_and_test_rust_project(data, tools)
                # Higher reward for passing tests
                score = 2.0 if cargo_results['test_passed'] else 0.0
            results.append(score)
        return results

    def _has_code_block(self, content: str) -> bool:
        """Check if response has proper code block structure"""
        return bool(self.evaluator.extract_rust_code(content) and "fn " in content)

    def _has_test_block(self, content: str) -> bool:
        """Check if response has test block"""
        return bool(self.evaluator.extract_test_code(content))


def create_dataset(path: str, system_prompt: str) -> Dataset:
    """Create dataset for training"""
    data = load_dataset("parquet", data_files={"train": path})["train"]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': x['rust_prompt']}
        ],
        "test_list": x.get('rust_test_list', [])
    })
    return data


def setup_model_and_tokenizer(
    model_name: str, 
    use_peft: bool = True, 
    use_gpu: bool = True
) -> tuple:
    """Set up model and tokenizer with optional PEFT"""
    # Limit to a single GPU to avoid accelerate auto-sharding across cuda:0/cuda:1
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    # Device setup
    device = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and model (avoid accelerate device_map to keep model on a single device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    ).to(device)

    # Setup PEFT if requested
    peft_config = None
    if use_peft:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )

    model.enable_input_require_grads()
    
    # Print parameter counts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || "
          f"Trainable%: {100 * trainable_params / all_params:.2f}")
    
    return model, tokenizer, peft_config


def train_rust_coder(
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    train_file: str = "cargo_test_passed_train.parquet",
    output_dir: str = "outputs",
    use_peft: bool = True,
    use_gpu: bool = True,
    num_generations: int = 4,
    batch_size: int = 1,
    learning_rate: float = 5e-6,
    num_epochs: int = 1,
    save_steps: int = 100
):
    """Main training function"""
    
    print(f"Training {model_name} on {train_file}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup dataset
    train_dataset = create_dataset(train_file, SYSTEM_PROMPT)
    print(f"Dataset size: {len(train_dataset)}")
    
    # Setup model
    model, tokenizer, peft_config = setup_model_and_tokenizer(
        model_name, use_peft, use_gpu
    )
    
    # Setup reward functions
    evaluator = RustCodeEvaluator()
    rewards = RewardFunctions(evaluator)
    
    reward_functions = [
        rewards.create_reward_function("cargo_build", rewards.cargo_build_reward),
        rewards.create_reward_function("cargo_clippy", rewards.cargo_clippy_reward),
        rewards.create_reward_function("cargo_test", rewards.cargo_test_reward),
        rewards.create_reward_function("non_empty", rewards.non_empty_reward),
        rewards.create_reward_function("test_block", rewards.test_block_reward),
        rewards.create_reward_function("test_asserts", rewards.test_asserts_reward),
    ]
    
    # Training configuration
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        num_generations=num_generations,
        max_prompt_length=256,
        max_completion_length=786,
        num_train_epochs=num_epochs,
        save_steps=save_steps,
        save_total_limit=1,
        max_grad_norm=0.1,
        log_on_each_node=False,
        optim="adamw_torch"
    )
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    print("Training completed!")
    return trainer


if __name__ == "__main__":
    # Example usage
    trainer = train_rust_coder(
        model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        train_file="cargo_test_passed_train.parquet",
        output_dir="rust_coder_outputs",
        use_peft=True,
        use_gpu=True,
        num_generations=4,
        batch_size=1,
        learning_rate=5e-6,
        num_epochs=1,
        save_steps=50
    )