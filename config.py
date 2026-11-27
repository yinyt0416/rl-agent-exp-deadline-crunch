"""MVP environment configuration.

All values here are provisional defaults. Override them by passing an EnvConfig
instance to DDLEnv / generate_dataset, or load from JSON with EnvConfig.from_json.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class EnvConfig:
    # Budget
    total_time_minutes: int = 120
    total_compute: int = 10

    # Cost of one action: (time in minutes, compute units)
    action_costs: dict = field(default_factory=lambda: {
        "debug": {"time": 10, "compute": 1},
        "train": {"time": 25, "compute": 3},
        "run_normal": {"time": 15, "compute": 2},
        "run_hard": {"time": 20, "compute": 2},
        "organize": {"time": 10, "compute": 0},
        "write_report": {"time": 15, "compute": 0},
        "submit": {"time": 0, "compute": 0},
    })

    # Random events
    train_failure_prob_no_debug: float = 0.35
    train_failure_prob_with_debug: float = 0.05
    experiment_failure_prob: float = 0.10

    # Pre-generated paired A/B results
    score_base: float = 0.58
    score_bonus_if_a_better: float = 0.14
    result_noise_std: float = 0.12
    results_per_track: int = 12
    coverage_target: int = 2

    # Delivery requirements
    model_required: bool = True
    report_required: bool = True
    organize_required: bool = True

    # Final quality weights (must sum to 1.0)
    weights: dict = field(default_factory=lambda: {
        "truth": 0.60,
        "model": 0.15,
        "report": 0.10,
        "organize": 0.05,
        "coverage": 0.10,
    })

    # Training-only auxiliary reward; never used in formal comparison.
    auxiliary_progress_reward: float = 0.05
    illegal_action_penalty: float = 0.10

    # Reward rounding for stable logging
    reward_decimals: int = 4

    def cost_of(self, action_name):
        return self.action_costs[action_name]

    def to_dict(self):
        return asdict(self)

    def to_json(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
