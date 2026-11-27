"""MVP self-check: reproducibility, scenario coverage, interface and lifecycle."""

import json
from pathlib import Path

from config import EnvConfig
from ddl_env import DDLEnv
from evaluation import derive_claims, evaluate_policy
from scenario_generator import TRACKS, TRUTH_COMBOS, generate_scenario, load_scenarios

DATA_DIR = Path(__file__).resolve().parent / "data"


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("  ok -", message)


def main():
    cfg = EnvConfig()

    print("[1] dataset manifest")
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    counts = {split: info["count"] for split, info in manifest["splits"].items()}
    check(counts == {"train": 24, "val": 8, "test": 12}, f"split counts {counts}")
    for split in ("train", "val", "test"):
        check(len(load_scenarios(split, DATA_DIR)) == counts[split], f"{split}.jsonl loads")

    print("[2] four hidden-truth combinations")
    test = load_scenarios("test", DATA_DIR)
    combos = {
        (s["truth"]["normal"], s["truth"]["hard"])
        for s in test
    }
    expected = {(c["normal"], c["hard"]) for c in TRUTH_COMBOS}
    check(combos == expected, f"test covers all 4 combos: {sorted(combos)}")

    print("[3] determinism")
    a = generate_scenario("check-0000", "test", 0, cfg)
    b = generate_scenario("check-0000", "test", 0, cfg)
    check(a == b, "same index + seed produces identical scenario")
    again = generate_scenario("other-0000", "train", 0, cfg)
    check(a["truth"] == again["truth"], "truth mapping is stable across splits")

    print("[4] interface and observation")
    scenario = test[0]
    env = DDLEnv(scenario, config=cfg)
    obs = env.reset()
    check("truth" not in obs, "hidden truth absent from observation")
    check(obs["done"] is False, "episode starts unfinished")
    check("submit" in env.valid_actions(), "submit is always legal")
    if not obs["progress"]["model_ready"]:
        check(not any(x.startswith("run_") for x in env.valid_actions()),
              "experiment actions require a ready model")

    before = env.observe()
    _obs, reward, done, info = env.step("__not_an_action__")
    check(info.get("illegal") is True, "illegal action is rejected")
    check(reward < 0 and not done, "illegal action penalized without ending episode")
    check(env.observe() == before, "illegal action leaves state unchanged")

    print("[5] lifecycle and terminal result")
    env = DDLEnv(scenario, config=cfg)
    env.reset()
    steps = 0
    final_info = None
    while not env.observe()["done"] and steps < 200:
        legal = env.valid_actions()
        _obs, _reward, _done, final_info = env.step(legal[0])
        steps += 1
    check(env.done and steps < 200, f"episode terminates in {steps} steps")
    check(final_info is not None and "final_result" in final_info, "terminal info carries final result")
    check(env.last_result is not None, "final result produced")
    quality = env.last_result["quality"]
    check(0.0 <= quality <= 1.0, f"quality in [0,1]: {quality}")
    check("truth" in env.last_result, "truth revealed after episode end")

    print("[6] auxiliary reward is training-only")
    env = DDLEnv(scenario, config=cfg, use_aux_reward=True)
    env.reset()
    _obs, reward, done, info = env.step("debug")
    check(not done and reward > 0 and info["aux_reward"] > 0,
          "progress step emits auxiliary reward when enabled")

    print("[7] random-policy batch run")
    import random

    class RandomPolicy:
        def __init__(self):
            self.rng = random.Random(42)

        def __call__(self, observation):
            # Derive legality from the observation using the shared rules.
            legal = ["debug", "train", "run_normal", "run_hard", "organize", "write_report", "submit"]
            p = observation["progress"]
            cost = cfg.action_costs
            def affordable(name):
                return (observation["remaining_time"] >= cost[name]["time"]
                        and observation["remaining_compute"] >= cost[name]["compute"])
            legal = [
                name for name in legal
                if affordable(name)
                and not (name == "debug" and p["debug_done"])
                and not (name == "train" and p["model_ready"])
                and not (name.startswith("run_") and not p["model_ready"])
                and not (name == "organize" and p["results_organized"])
                and not (name == "write_report" and p["report_written"])
                and not (name.startswith("run_") and p["tests_run"][name[4:]] >= cfg.results_per_track)
            ]
            name = self.rng.choice(legal)
            if name == "submit":
                return {"name": "submit", "claims": derive_claims(observation["observed_results"])}
            return name

    metrics = evaluate_policy(RandomPolicy(), test, cfg)
    check(metrics["n_scenarios"] == len(test), "all scenarios evaluated")
    check(0.0 <= metrics["avg_quality"] <= 1.0, f"avg quality {metrics['avg_quality']}")
    check(0.0 <= metrics["effective_delivery_rate"] <= 1.0,
          f"effective delivery rate {metrics['effective_delivery_rate']}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "records"},
                     ensure_ascii=False, indent=2))
    print("SELF-CHECK OK")


if __name__ == "__main__":
    main()
