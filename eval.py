#!/usr/bin/env python3
"""
Rust Code Evaluation Script

This script evaluates Rust code predictions by running cargo build, clippy, and test
commands on generated code and saving the results.
"""

import os
import subprocess
import shutil
import argparse
from pathlib import Path
from uuid import uuid4
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


class RustTool:
    """Tool for running Rust cargo commands with timeout handling"""
    
    def __init__(self, name: str, timeout: int = 10):
        self.name = name
        self.timeout = timeout

    def run(self, results: Dict[str, Any], project_dir: Path) -> Dict[str, Any]:
        """Run cargo command and update results dictionary"""
        try:
            print(f"Running: cargo {self.name} in {project_dir}")
            result = subprocess.run(
                ["cargo", self.name, "--quiet"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            success = result.returncode == 0
            results[f'{self.name}_passed'] = success
            results[f'{self.name}_stderr'] = result.stderr
            
            if not success:
                print(f"cargo {self.name} failed: {result.stderr}")
            else:
                print(f"cargo {self.name} passed")
                
        except subprocess.TimeoutExpired:
            results[f'{self.name}_passed'] = False
            results[f'{self.name}_stderr'] = f"cargo {self.name} timeout ({self.timeout}s)"
            print(f"cargo {self.name} timed out")
        except Exception as e:
            results[f'{self.name}_passed'] = False
            results[f'{self.name}_stderr'] = f"cargo {self.name} error: {e}"
            print(f"cargo {self.name} error: {e}")
            
        return results


class RustCodeEvaluator:
    """Evaluates Rust code by setting up temporary projects and running cargo commands"""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.test_dir = self.output_dir / "tests"
        self.test_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def extract_rust_code(rust_code: str) -> str:
        """Extract Rust code from markdown code blocks or return as-is"""
        if "```rust" in rust_code:
            code = rust_code.split("```rust")[-1]
            code = code.split("```")[0]
            return code.strip()
        return rust_code

    @staticmethod
    def get_rust_template() -> str:
        """Get the Rust main.rs template"""
        return """
// {code}

fn main() {
    println!("Hello, world!");
}
"""

    @staticmethod
    def get_cargo_toml() -> str:
        """Get the Cargo.toml template"""
        return """
[package]
name = "rust-program"
version = "0.1.0"
edition = "2021"

[dependencies]
"""

    def setup_and_test_rust_project(self, row: Dict[str, Any], tools: List[RustTool]) -> Dict[str, Any]:
        """
        Set up a temporary Rust project and run evaluation tools
        
        Args:
            row: Dictionary containing 'response' field with Rust code
            tools: List of RustTool objects to run
            
        Returns:
            Dictionary with test results and metadata
        """
        # Create unique temporary project directory
        project_dir = self.test_dir / f"temp_rust_project_{uuid4()}"
        project_dir_src = project_dir / "src"
        project_dir_src.mkdir(parents=True, exist_ok=True)

        try:
            # Prepare Rust code from template
            template = self.get_rust_template()
            rust_code = self.extract_rust_code(row['response'])
            filled_template = template.replace("// {code}", rust_code)

            # Write main.rs
            main_rs_path = project_dir_src / "main.rs"
            with open(main_rs_path, "w") as f:
                f.write(filled_template)

            # Write Cargo.toml
            cargo_file_path = project_dir / "Cargo.toml"
            with open(cargo_file_path, "w") as f:
                f.write(self.get_cargo_toml())

            # Initialize results with template
            results = {
                'template': filled_template,
                'rust_code_extracted': rust_code
            }

            # Run all tools
            for tool in tools:
                results = tool.run(results, project_dir)

            return results

        finally:
            # Always clean up the temporary project
            if project_dir.exists():
                shutil.rmtree(project_dir)


class ResultsAnalyzer:
    """Analyzes and visualizes evaluation results"""
    
    @staticmethod
    def calculate_pass_rates(df: pd.DataFrame, tools: List[RustTool]) -> Dict[str, float]:
        """Calculate pass rates for each tool"""
        pass_rates = {}
        for tool in tools:
            col_name = f'{tool.name}_passed'
            if col_name in df.columns:
                pass_rate = df[col_name].mean() * 100
                pass_rates[tool.name] = pass_rate
        return pass_rates

    @staticmethod
    def plot_results(df: pd.DataFrame, tools: List[RustTool], save_path: str = None) -> None:
        """Create visualization of evaluation results"""
        num_tools = len(tools)
        fig, axes = plt.subplots(1, num_tools, figsize=(4 * num_tools, 4))
        
        if num_tools == 1:
            axes = [axes]

        # Retro color palette
        colors = ['#6fcb9f', '#fb2e01']  # Green for pass, Red for fail
        
        for i, tool in enumerate(tools):
            col_name = f'{tool.name}_passed'
            if col_name not in df.columns:
                continue
                
            # Count passes/fails
            counts = df[col_name].value_counts()
            num_passed = counts.get(True, 0)
            num_failed = counts.get(False, 0)
            total = num_passed + num_failed
            
            if total == 0:
                continue
                
            percentage = (num_passed / total) * 100
            
            # Plot bar chart
            ax = axes[i]
            ax.bar(['Passed', 'Failed'], [num_passed, num_failed], color=colors)
            ax.set_title(f'{tool.name.title()}: {num_passed}/{total} = {percentage:.1f}%')
            ax.set_ylabel('Count')
            
            # Add value labels on bars
            for j, v in enumerate([num_passed, num_failed]):
                if v > 0:
                    ax.text(j, v + total * 0.01, str(v), ha='center', va='bottom')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()

    @staticmethod
    def print_summary(df: pd.DataFrame, tools: List[RustTool]) -> None:
        """Print summary statistics"""
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        total_samples = len(df)
        print(f"Total samples evaluated: {total_samples}")
        
        # Individual tool results
        for tool in tools:
            col_name = f'{tool.name}_passed'
            if col_name in df.columns:
                num_passed = df[col_name].sum()
                pass_rate = (num_passed / total_samples) * 100
                print(f"{tool.name.title()} pass rate: {num_passed}/{total_samples} = {pass_rate:.1f}%")
        
        # Overall pass rate (all tools must pass)
        if len(tools) > 1:
            all_passed = df[[f'{tool.name}_passed' for tool in tools if f'{tool.name}_passed' in df.columns]].all(axis=1)
            overall_pass_rate = all_passed.mean() * 100
            overall_passed = all_passed.sum()
            print(f"Overall pass rate (all tools): {overall_passed}/{total_samples} = {overall_pass_rate:.1f}%")


def evaluate_rust_predictions(
    input_file: str,
    output_file: str,
    tools: List[RustTool] = None,
    max_rows: int = -1,
    save_plot: bool = True
) -> pd.DataFrame:
    """
    Main evaluation function
    
    Args:
        input_file: Path to parquet file with predictions
        output_file: Path to save results
        tools: List of RustTool objects (default: build, clippy, test)
        max_rows: Maximum number of rows to evaluate (-1 for all)
        save_plot: Whether to save visualization plot
        
    Returns:
        DataFrame with evaluation results
    """
    
    # Default tools if none provided
    if tools is None:
        tools = [
            RustTool("build", timeout=10),
            RustTool("clippy", timeout=10),
            RustTool("test", timeout=10)
        ]
    
    print(f"Loading predictions from: {input_file}")
    df = pd.read_parquet(input_file)
    
    # Limit rows if specified
    if max_rows > 0:
        df = df.head(max_rows)
        print(f"Evaluating first {max_rows} rows")
    else:
        print(f"Evaluating all {len(df)} rows")
    
    # Initialize evaluator
    evaluator = RustCodeEvaluator()
    results = []
    
    # Track progress
    total_passed = 0
    
    print("\nStarting evaluation...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        # Run evaluation
        test_results = evaluator.setup_and_test_rust_project(row.to_dict(), tools)
        test_results['idx'] = idx
        
        # Merge original row with test results
        row_dict = row.to_dict()
        row_dict.update(test_results)
        results.append(row_dict)
        
        # Count successes
        num_passed = sum(test_results[f'{tool.name}_passed'] for tool in tools)
        all_passed = num_passed == len(tools)
        
        if all_passed:
            total_passed += 1
        
        # Print progress every 10 samples
        if (idx + 1) % 10 == 0:
            accuracy = (total_passed / (idx + 1)) * 100
            print(f"Progress: {idx + 1}/{len(df)} | Overall pass rate: {total_passed}/{idx + 1} = {accuracy:.1f}%")
        
        # Save intermediate results every 100 samples
        if (idx + 1) % 100 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_parquet(output_file, index=False)
            print(f"Intermediate results saved to: {output_file}")
    
    # Create final results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save final results
    results_df.to_parquet(output_file, index=False)
    print(f"\nFinal results saved to: {output_file}")
    
    # Analyze and visualize results
    analyzer = ResultsAnalyzer()
    analyzer.print_summary(results_df, tools)
    
    if save_plot:
        plot_path = output_file.replace('.parquet', '_results.png')
        analyzer.plot_results(results_df, tools, plot_path)
    
    return results_df


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Evaluate Rust code predictions")
    parser.add_argument("input_file", help="Input parquet file with predictions")
    parser.add_argument("output_file", help="Output parquet file for results")
    parser.add_argument("--max-rows", type=int, default=-1, 
                       help="Maximum number of rows to evaluate (-1 for all)")
    parser.add_argument("--timeout", type=int, default=10,
                       help="Timeout for cargo commands in seconds")
    parser.add_argument("--no-plot", action="store_true",
                       help="Skip saving visualization plot")
    parser.add_argument("--tools", nargs="+", default=["build", "clippy", "test"],
                       help="Cargo tools to run (default: build clippy test)")
    
    args = parser.parse_args()
    
    # Create tools
    tools = [RustTool(tool_name, timeout=args.timeout) for tool_name in args.tools]
    
    # Run evaluation
    results_df = evaluate_rust_predictions(
        input_file=args.input_file,
        output_file=args.output_file,
        tools=tools,
        max_rows=args.max_rows,
        save_plot=not args.no_plot
    )
    
    print(f"\nEvaluation complete! Results shape: {results_df.shape}")


if __name__ == "__main__":
    main()