from __future__ import annotations

import sys
from pathlib import Path

import torch

example_dir = Path(__file__).resolve().parent
tests_dir = example_dir.parent
runtime_dir = tests_dir.parent
sys.path.insert(0, str(runtime_dir))

from cuda_tile_runner import launch


scratch = torch.zeros(1, dtype=torch.float32, device="cuda")
launch(example_dir / "token-example.tileirbc", scratch)
print(f"final scratch = {scratch.item()}")
