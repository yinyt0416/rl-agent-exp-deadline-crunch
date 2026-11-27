# rl-agent-exp-deadline-crunch — 数据集与模拟环境 MVP

What to do if experiments cannot be finished before the deadline？ (A RL agent)

公共 DDL 环境与场景数据集，供五种策略共用。接口与默认配置见 `对齐文档.md`。

## 运行

```bash
python generate_data.py            # 重新生成 data/ 下固定场景（一般不需要）
python self_check.py               # 自检：可复现性、四类场景、接口、生命周期
python smoke_test.py               # 随机策略单局回放 + test split 批量评测
```

无第三方依赖，Python 3.8+ 标准库即可。

## 接入方式

```python
from config import EnvConfig
from ddl_env import DDLEnv
from scenario_generator import load_scenarios

scenario = load_scenarios("test")[0]
env = DDLEnv(scenario, config=EnvConfig(), use_aux_reward=False)  # 训练时可为 True
obs = env.reset()
action = env.valid_actions()[0]     # 或 {"name": "submit", "claims": {...}}
obs, reward, done, info = env.step(action)
```

用 `evaluation.evaluate_policy(policy, scenarios)` 做批量比较；`policy(observation) -> action`。

## 目录

- `config.py` / `default_config.json`：全部暂定默认参数
- `ddl_env.py`：环境主体与统一接口
- `scenario_generator.py`：四类真值场景与带噪声 A/B 结果串生成
- `evaluation.py`：终局评分与批量评测
- `generate_data.py` / `self_check.py` / `smoke_test.py`：脚本
- `data/`：固定 train / val / test 场景与 manifest
- `对齐文档.md`：接口、配置、评分口径与待确认事项
