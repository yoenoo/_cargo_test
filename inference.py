#!/usr/bin/env python3
"""
Rust Code Inference Script

This script runs inference on a dataset using a trained model to generate Rust code predictions.
Supports both local files and remote datasets, with configurable generation parameters.
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from tqdm import tqdm


# System prompt for Rust code generation
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


class RustCodeInference:
    """Handles model loading and inference for Rust code generation"""
    
    def __init__(
        self,
        model_name: str,
        use_gpu: bool = True,
        system_prompt: str = SYSTEM_PROMPT
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.device = self._setup_device(use_gpu)
        self.tokenizer = None
        self.model = None
        self.streamer = None
        
    def _setup_device(self, use_gpu: bool) -> str:
        """Determine the best device to use"""
        if use_gpu and torch.cuda.is_available():
            device = "cuda"
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("Using CPU")
        return device
    
    def load_model(self, verbose: bool = True) -> None:
        """Load the model and tokenizer"""
        if verbose:
            print(f"Loading model: {self.model_name}")
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        ).to(self.device)
        
        # Setup text streamer for real-time output (optional)
        self.streamer = TextStreamer(self.tokenizer, skip_special_tokens=True)
        
        if verbose:
            print(f"Model loaded on device: {self.device}")
    
    def format_prompt(self, rust_prompt: str) -> str:
        """Format the input prompt using the chat template"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": rust_prompt}
        ]
        
        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )
        
        return formatted_prompt
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: bool = True,
        stream_output: bool = False
    ) -> Dict[str, str]:
        """Generate a response for a single prompt"""
        
        # Format the prompt
        formatted_prompt = self.format_prompt(prompt)
        
        # Tokenize input
        inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Generate with optional streaming
        streamer = self.streamer if stream_output else None
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                streamer=streamer,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode outputs
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_only = self.tokenizer.decode(
            outputs[0][len(inputs[0]):], 
            skip_special_tokens=True
        )
        
        return {
            "formatted_prompt": formatted_prompt,
            "full_response": full_response,
            "response": response_only.strip()
        }


def load_dataset(dataset_path: str, oxen_repo: Optional[str] = None) -> pd.DataFrame:
    """Load dataset from local file or download from Oxen repository"""
    
    if not os.path.exists(dataset_path):
        if oxen_repo:
            try:
                from oxen import RemoteRepo
                print(f"Downloading {dataset_path} from {oxen_repo}")
                repo = RemoteRepo(oxen_repo)
                repo.download(dataset_path)
            except ImportError:
                raise ImportError("oxen package required for remote dataset loading. Install with: pip install oxen")
            except Exception as e:
                raise RuntimeError(f"Failed to download dataset: {e}")
        else:
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    else:
        print(f"Loading dataset from: {dataset_path}")
    
    df = pd.read_parquet(dataset_path)
    print(f"Dataset loaded with {len(df)} samples")
    return df


def save_results(
    results: List[Dict[str, Any]], 
    output_path: str,
    intermediate: bool = False
) -> None:
    """Save results to parquet file"""
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame and save
    result_df = pd.DataFrame(results)
    result_df.to_parquet(output_path, index=False)
    
    status = "Intermediate" if intermediate else "Final"
    print(f"{status} results saved to: {output_path}")


def run_inference(
    model_name: str,
    dataset_path: str,
    output_path: str,
    oxen_repo: Optional[str] = None,
    use_gpu: bool = True,
    max_samples: int = -1,
    save_every: int = 10,
    stream_output: bool = False,
    generation_config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Run inference on the entire dataset
    
    Args:
        model_name: HuggingFace model name or local path
        dataset_path: Path to dataset parquet file
        output_path: Path to save results
        oxen_repo: Optional Oxen repository for remote datasets
        use_gpu: Whether to use GPU if available
        max_samples: Maximum number of samples to process (-1 for all)
        save_every: Save intermediate results every N samples
        stream_output: Whether to stream generation output to console
        generation_config: Optional generation parameters
        
    Returns:
        DataFrame with results
    """
    
    # Default generation config
    if generation_config is None:
        generation_config = {
            "max_new_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.9,
            "do_sample": True
        }
    
    # Load dataset
    df = load_dataset(dataset_path, oxen_repo)
    
    # Limit samples if specified
    if max_samples > 0:
        df = df.head(max_samples)
        print(f"Processing first {max_samples} samples")
    
    # Initialize inference engine
    print(f"Initializing inference with model: {model_name}")
    inference = RustCodeInference(model_name, use_gpu)
    inference.load_model()
    
    # Process samples
    results = []
    
    print(f"\nStarting inference on {len(df)} samples...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        
        # Generate response
        generation_result = inference.generate_response(
            prompt=row['rust_prompt'],
            stream_output=stream_output,
            **generation_config
        )
        
        # Store result
        result = {
            "task_id": row.get('task_id', f"task_{index}"),
            "prompt": row['rust_prompt'],
            "test_list": row.get('rust_test_list', []),
            "input": generation_result["formatted_prompt"],
            "full_response": generation_result["full_response"],
            "response": generation_result["response"]
        }
        results.append(result)
        
        # Save intermediate results
        if len(results) % save_every == 0:
            save_results(results, output_path, intermediate=True)
        
        # Optional: print response for debugging
        if stream_output:
            print(f"\n{'='*50}")
            print(f"Sample {index + 1}/{len(df)}")
            print(f"Prompt: {row['rust_prompt'][:100]}...")
            print(f"Response: {generation_result['response'][:200]}...")
            print(f"{'='*50}\n")
    
    # Save final results
    save_results(results, output_path, intermediate=False)
    
    # Create final DataFrame
    results_df = pd.DataFrame(results)
    
    print(f"\nInference complete!")
    print(f"Processed {len(results)} samples")
    print(f"Results saved to: {output_path}")
    
    return results_df


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Run Rust code inference")
    
    # Required arguments
    parser.add_argument("model_name", help="HuggingFace model name or local path")
    parser.add_argument("dataset_path", help="Path to dataset parquet file")
    parser.add_argument("output_path", help="Path to save results")
    
    # Optional arguments
    parser.add_argument("--oxen-repo", help="Oxen repository for remote datasets")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--max-samples", type=int, default=-1, 
                       help="Maximum samples to process (-1 for all)")
    parser.add_argument("--save-every", type=int, default=10,
                       help="Save intermediate results every N samples")
    parser.add_argument("--stream", action="store_true",
                       help="Stream generation output to console")
    
    # Generation parameters
    parser.add_argument("--max-tokens", type=int, default=1024,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--deterministic", action="store_true",
                       help="Use deterministic generation (do_sample=False)")
    
    args = parser.parse_args()
    
    # Prepare generation config
    generation_config = {
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": not args.deterministic
    }
    
    # Run inference
    results_df = run_inference(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        oxen_repo=args.oxen_repo,
        use_gpu=not args.no_gpu,
        max_samples=args.max_samples,
        save_every=args.save_every,
        stream_output=args.stream,
        generation_config=generation_config
    )
    
    print(f"Final results shape: {results_df.shape}")


if __name__ == "__main__":
    main()