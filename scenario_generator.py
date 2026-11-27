"""Deterministic scenario / dataset generation for the DDL environment.

A scenario fixes everything the agents cannot control: the hidden truth for the
two tracks, the pre-generated noisy A/B result streams, the opening state and
the seed used for random events. All policies see the same scenario.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from config import EnvConfig

TRACKS = ("normal", "hard")
TRUTH_COMBOS = (
    {"normal": False, "hard": False},
    {"normal": True, "hard": False},
    {"normal": False, "hard": True},
    {"normal": True, "hard": True},
)
OPENINGS = (
    {"model_ready": True, "debug_done": True, "report_written": False, "results_organized": False},
    {"model_ready": False, "debug_done": False, "report_written": True, "results_organized": False},
    {"model_ready": False, "debug_done": False, "report_written": False, "results_organized": True},
    {"model_ready": False, "debug_done": False, "report_written": False, "results_organized": False},
)
SEED_BASES = {"train": 1000, "val": 2000, "test": 3000}


def make_result_stream(truth_for_track, seed, config):
    """Pre-generate one paired A/B result stream with Gaussian noise."""
    rng = random.Random(seed)
    stream = []
    for _ in range(config.results_per_track):
        a = config.score_base + (config.score_bonus_if_a_better if truth_for_track else 0.0)
        b = config.score_base
        a += rng.gauss(0.0, config.result_noise_std)
        b += rng.gauss(0.0, config.result_noise_std)
        stream.append({
            "a": round(max(0.0, min(1.0, a)), 4),
            "b": round(max(0.0, min(1.0, b)), 4),
        })
    return stream


def generate_scenario(scenario_id, split, index, config=None):
    config = config or EnvConfig()
    seed = SEED_BASES[split] + index
    truth = dict(TRUTH_COMBOS[index % len(TRUTH_COMBOS)])
    opening_rng = random.Random(seed + 17)
    opening = dict(OPENINGS[opening_rng.randrange(len(OPENINGS))])
    results = {
        track: make_result_stream(truth[track], seed + (1 if track == "normal" else 2), config)
        for track in TRACKS
    }
    return {
        "scenario_id": scenario_id,
        "split": split,
        "index": index,
        "seed": seed,
        "truth": truth,
        "initial_progress": opening,
        "results": results,
    }


def generate_dataset(output_dir="data", n_train=24, n_val=8, n_test=12, config=None):
    """Write data/{train,val,test}.jsonl and data/manifest.json. Returns manifest."""
    config = config or EnvConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"config": config.to_dict(), "splits": {}}
    for split, count in (("train", n_train), ("val", n_val), ("test", n_test)):
        scenarios = [
            generate_scenario(f"{split}-{i:04d}", split, i, config)
            for i in range(count)
        ]
        path = out / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in scenarios),
            encoding="utf-8",
        )
        manifest["splits"][split] = {
            "count": count,
            "file": path.name,
            "seed_range": [SEED_BASES[split], SEED_BASES[split] + count - 1],
            "scenario_ids": [s["scenario_id"] for s in scenarios],
        }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def load_scenarios(split, data_dir="data"):
    path = Path(data_dir) / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_all_scenarios(data_dir="data"):
    result = []
    for split in ("train", "val", "test"):
        result.extend(load_scenarios(split, data_dir))
    return result
