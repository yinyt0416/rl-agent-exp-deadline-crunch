"""MVP DDL environment.

Implements the shared interface used by all five strategies:
    reset() -> observation
    step(action) -> (observation, reward, done, info)
    valid_actions() -> list[str]
    observe() -> observation

The hidden truth is never exposed in observations. It is revealed in info only
after the episode ends, for replay and evaluation.
"""

from __future__ import annotations

import copy
import random

from config import EnvConfig
from evaluation import derive_claims, final_result
from scenario_generator import TRACKS

PROGRESS_EVENTS = {
    "debug_done", "train_success", "first_result_normal", "first_result_hard",
    "results_organized", "report_written",
}


class DDLEnv:
    def __init__(self, scenario, config=None, use_aux_reward=False):
        self.scenario = scenario
        self.config = config or EnvConfig()
        self.use_aux_reward = use_aux_reward
        self.reset()

    def reset(self):
        cfg = self.config
        self.time_left = cfg.total_time_minutes
        self.compute_left = cfg.total_compute
        self.progress = copy.deepcopy(self.scenario["initial_progress"])
        self.progress["tests_run"] = {track: 0 for track in TRACKS}
        self.observed = {track: [] for track in TRACKS}
        self.done = False
        self.submitted = False
        self.on_time = True
        self.steps = 0
        self.last_result = None
        self.rng = random.Random(self.scenario["seed"] + 7777)
        return self.observe()

    def observe(self):
        return {
            "remaining_time": self.time_left,
            "remaining_compute": self.compute_left,
            "progress": {
                "debug_done": self.progress["debug_done"],
                "model_ready": self.progress["model_ready"],
                "results_organized": self.progress["results_organized"],
                "report_written": self.progress["report_written"],
                "tests_run": dict(self.progress["tests_run"]),
            },
            "observed_results": {
                track: [dict(item) for item in self.observed[track]] for track in TRACKS
            },
            "done": self.done,
        }

    def _affordable(self, action_name):
        cost = self.config.action_costs[action_name]
        return (
            self.time_left >= cost["time"]
            and self.compute_left >= cost["compute"]
        )

    def valid_actions(self):
        if self.done:
            return []
        actions = []
        if not self.progress["debug_done"] and self._affordable("debug"):
            actions.append("debug")
        if not self.progress["model_ready"] and self._affordable("train"):
            actions.append("train")
        for track in TRACKS:
            stream = self.scenario["results"][track]
            if (
                self.progress["model_ready"]
                and self.progress["tests_run"][track] < len(stream)
                and self._affordable(f"run_{track}")
            ):
                actions.append(f"run_{track}")
        if not self.progress["results_organized"] and self._affordable("organize"):
            actions.append("organize")
        if not self.progress["report_written"] and self._affordable("write_report"):
            actions.append("write_report")
        actions.append("submit")
        return actions

    @staticmethod
    def _action_name(action):
        if isinstance(action, str):
            return action, {}
        if isinstance(action, dict):
            return action.get("name"), action
        return None, {}

    def _finish(self, claims, submitted, on_time):
        self.done = True
        self.submitted = submitted
        self.on_time = on_time
        self.last_result = final_result(
            self.progress,
            claims,
            self.scenario["truth"],
            self.observed,
            self.config,
            submitted=submitted,
            on_time=on_time,
        )
        return self.last_result["quality"]

    def step(self, action):
        if self.done:
            raise ValueError("Episode is already done. Call reset() first.")

        name, params = self._action_name(action)
        if name not in self.valid_actions():
            return (
                self.observe(),
                -self.config.illegal_action_penalty,
                False,
                {"illegal": True, "reason": f"action unavailable: {name}"},
            )

        cfg = self.config
        cost = cfg.action_costs[name]
        self.time_left -= cost["time"]
        self.compute_left -= cost["compute"]
        self.steps += 1
        event = name
        aux = 0.0

        if name == "debug":
            self.progress["debug_done"] = True
            event = "debug_done"

        elif name == "train":
            fail_prob = (
                cfg.train_failure_prob_with_debug
                if self.progress["debug_done"]
                else cfg.train_failure_prob_no_debug
            )
            if self.rng.random() < fail_prob:
                event = "train_failed"
            else:
                self.progress["model_ready"] = True
                event = "train_success"

        elif name in ("run_normal", "run_hard"):
            track = name.replace("run_", "")
            if self.rng.random() < cfg.experiment_failure_prob:
                event = "experiment_failed"
            else:
                index = self.progress["tests_run"][track]
                result = self.scenario["results"][track][index]
                self.observed[track].append(dict(result))
                self.progress["tests_run"][track] += 1
                event = f"first_result_{track}" if index == 0 else "result_added"

        elif name == "organize":
            self.progress["results_organized"] = True
            event = "results_organized"

        elif name == "write_report":
            self.progress["report_written"] = True
            event = "report_written"

        elif name == "submit":
            claims = params.get("claims") or derive_claims(self.observed)
            claims = {track: bool(claims[track]) for track in TRACKS}
            return self._finish_terminal(claims, submitted=True, on_time=True, event=event)

        if event in PROGRESS_EVENTS:
            aux = cfg.auxiliary_progress_reward

        forced = self.time_left <= 0 or self.compute_left <= 0
        if forced:
            return self._finish_terminal(
                derive_claims(self.observed),
                submitted=False,
                on_time=self.time_left > 0,
                event=event,
                aux=aux,
                forced=True,
            )

        reward = aux if self.use_aux_reward else 0.0
        return self.observe(), round(reward, cfg.reward_decimals), False, {
            "event": event,
            "aux_reward": aux,
            "cost": dict(cost),
            "step": self.steps,
        }

    def _finish_terminal(self, claims, submitted, on_time, event, aux=0.0, forced=False):
        quality = self._finish(claims, submitted, on_time)
        info = {
            "event": event,
            "aux_reward": aux,
            "step": self.steps,
            "submitted": submitted,
            "forced": forced,
            "claims": dict(claims),
            "final_result": self.last_result,
        }
        return self.observe(), round(quality, self.config.reward_decimals), True, info

    def render(self):
        progress = self.observe()["progress"]
        print(
            f"t={self.time_left:3d}  compute={self.compute_left:2d}  "
            f"model={progress['model_ready']}  tests="
            f"{progress['tests_run']}  report={progress['report_written']}  done={self.done}"
        )
