"""Run one detailed episode and a random-policy batch on the fixed test split."""

import argparse
import json
import random
from pathlib import Path

from config import EnvConfig
from ddl_env import DDLEnv
from evaluation import derive_claims, evaluate_policy
from scenario_generator import load_scenarios

BASE_DIR = Path(__file__).resolve().parent


class RandomPolicy:
    def __init__(self, seed=0, config=None):
        self.rng = random.Random(seed)
        self.config = config or EnvConfig()

    def __call__(self, observation):
        cfg = self.config
        cost = cfg.action_costs
        p = observation["progress"]

        def affordable(name):
            return (
                observation["remaining_time"] >= cost[name]["time"]
                and observation["remaining_compute"] >= cost[name]["compute"]
            )

        names = ("debug", "train", "run_normal", "run_hard", "organize", "write_report", "submit")
        legal = []
        for name in names:
            if not affordable(name):
                continue
            if name == "debug" and p["debug_done"]:
                continue
            if name == "train" and p["model_ready"]:
                continue
            if name.startswith("run_"):
                track = name[4:]
                if not p["model_ready"] or p["tests_run"][track] >= cfg.results_per_track:
                    continue
            if name == "organize" and p["results_organized"]:
                continue
            if name == "write_report" and p["report_written"]:
                continue
            legal.append(name)

        choice = self.rng.choice(legal)
        if choice == "submit":
            return {"name": "submit", "claims": derive_claims(observation["observed_results"])}
        return choice


def detailed_episode(scenario, config, seed=0):
    env = DDLEnv(scenario, config=config)
    env.reset()
    policy = RandomPolicy(seed=seed, config=config)
    print("\n--- detailed episode:", scenario["scenario_id"], "---")
    env.render()
    while not env.observe()["done"]:
        action = policy(env.observe())
        obs, reward, done, info = env.step(action)
        print(f"action={action} -> {info.get('event')} reward={reward} done={done}")
        env.render()
    print("final:", json.dumps(env.last_result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(BASE_DIR / "data"))
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    config = EnvConfig()
    scenarios = load_scenarios(args.split, args.data_dir)
    detailed_episode(scenarios[0], config)
    metrics = evaluate_policy(RandomPolicy(config=config), scenarios, config)
    print("\n--- batch metrics (random policy) ---")
    print(json.dumps({k: v for k, v in metrics.items() if k != "records"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
