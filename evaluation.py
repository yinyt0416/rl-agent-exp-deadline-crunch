"""Final scoring and batch evaluation.

The auxiliary reward is a training-only signal. Formal comparison uses
`quality` and `effective_delivery` from this module, not cumulative reward.
"""

from __future__ import annotations

from config import EnvConfig
from scenario_generator import TRACKS


def derive_claims(observed):
    """Default claim rule: A is better when the observed mean difference is positive."""
    claims = {}
    for track in TRACKS:
        items = observed.get(track, [])
        if not items:
            claims[track] = False
            continue
        diff = sum(item["a"] - item["b"] for item in items) / len(items)
        claims[track] = diff > 0
    return claims


def final_result(progress, claims, truth, observed, config=None, submitted=True, on_time=True):
    config = config or EnvConfig()
    correct = sum(1 for track in TRACKS if claims[track] == truth[track]) / len(TRACKS)

    tests = progress.get("tests_run", {})
    coverage = sum(
        min(tests.get(track, 0), config.coverage_target) / config.coverage_target
        for track in TRACKS
    ) / len(TRACKS)

    components = {
        "truth": correct,
        "model": 1.0 if progress.get("model_ready") else 0.0,
        "report": 1.0 if progress.get("report_written") else 0.0,
        "organize": 1.0 if progress.get("results_organized") else 0.0,
        "coverage": coverage,
    }
    quality = sum(config.weights[key] * value for key, value in components.items())

    artifacts_ok = True
    if config.model_required:
        artifacts_ok = artifacts_ok and progress.get("model_ready", False)
    if config.report_required:
        artifacts_ok = artifacts_ok and progress.get("report_written", False)
    if config.organize_required:
        artifacts_ok = artifacts_ok and progress.get("results_organized", False)

    effective = bool(submitted and on_time and artifacts_ok and correct == 1.0)
    return {
        "quality": round(quality, 4),
        "effective_delivery": effective,
        "components": {key: round(value, 4) for key, value in components.items()},
        "claims": dict(claims),
        "truth": dict(truth),
        "submitted": submitted,
        "on_time": on_time,
    }


def evaluate_policy(policy, scenarios, config=None, use_aux_reward=False):
    """Run a policy on every scenario and aggregate the five evaluation dimensions."""
    from ddl_env import DDLEnv

    config = config or EnvConfig()
    records = []
    for scenario in scenarios:
        env = DDLEnv(scenario, config=config, use_aux_reward=use_aux_reward)
        obs = env.reset()
        steps = 0
        while not obs["done"] and steps < 200:
            action = policy(obs)
            obs, _reward, done, info = env.step(action)
            steps += 1
            if done:
                break
        result = info.get("final_result")
        if result is None:
            result = final_result(
                env.progress,
                derive_claims(env.observed),
                env.scenario["truth"],
                env.observed,
                config,
                submitted=False,
                on_time=False,
            )
        records.append({
            "scenario_id": scenario["scenario_id"],
            "split": scenario["split"],
            "quality": result["quality"],
            "effective_delivery": result["effective_delivery"],
            "claims_correct": result["components"]["truth"],
            "task_completion": round(
                sum(result["components"][key] for key in ("model", "report", "organize", "coverage")) / 4,
                4,
            ),
            "time_used": config.total_time_minutes - env.time_left,
            "compute_used": config.total_compute - env.compute_left,
            "steps": env.steps,
        })
    n = len(records) or 1
    return {
        "n_scenarios": len(records),
        "effective_delivery_rate": round(sum(r["effective_delivery"] for r in records) / n, 4),
        "avg_quality": round(sum(r["quality"] for r in records) / n, 4),
        "avg_task_completion": round(sum(r["task_completion"] for r in records) / n, 4),
        "avg_time_used": round(sum(r["time_used"] for r in records) / n, 2),
        "avg_compute_used": round(sum(r["compute_used"] for r in records) / n, 2),
        "records": records,
    }
