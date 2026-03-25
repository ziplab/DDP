"""
VQA Challenge — Solver interface and evaluation runner.

This module provides the Solver abstract class and run() function.
Users should NOT edit this file. Instead, edit challenge_runner.py.
See examples/ for reference implementations.
"""

import json
from abc import ABC, abstractmethod

import pandas as pd
from tqdm import tqdm


class Solver(ABC):
    @abstractmethod
    def solve(self, image_path: str, prompt: str) -> int:
        """Return 0 (No), 1 (Yes), or -1 (failed)."""
        ...

    @abstractmethod
    def model_info(self) -> dict:
        """Return model name and generation parameters, e.g.:
        {"model": "gpt-4o", "parameters": {"temperature": 0.7, "top_p": 0.9, ...}}
        """
        ...


def run(solver: Solver, input_csv: str, output_txt: str, output_json: str = "model.json"):
    """Run the solver on every row of input_csv and write results to output_txt."""
    df = pd.read_csv(input_csv)
    assert "image_path" in df.columns and "prompt" in df.columns

    results = {}
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        results[idx] = solver.solve(row["image_path"], row["prompt"])

    # Write predictions
    with open(output_txt, "w") as f:
        f.write("\n".join(f"{idx} {results[idx]}" for idx in sorted(results.keys())))

    # Write model info JSON
    with open(output_json, "w") as f:
        json.dump(solver.model_info(), f, indent=2)

    answers = list(results.values())
    print(f"Results saved to {output_txt}")
    print(f"Model info saved to {output_json}")
    print(f"Total: {len(answers)} | Answer 0: {answers.count(0)} | Answer 1: {answers.count(1)} | Failed: {answers.count(-1)}")
