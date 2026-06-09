"""
Configuration Loading Module for ProcTHOR
  spatial-planning/config/load_config.py   ：
- YAML    + tasks/<task_id>/task.json   
- apply_task_by_name: max_steps = 10 + 2 * n（n=    golden actions  ）
"""
import json
import sys
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from actions.max_steps import compute_max_steps_from_n, derive_task_n


class ConfigLoader:
    """Configuration Loader for ProcTHOR."""

    DEFAULT_CONFIG = {
        "env": {
            "type": "procthor",
            "width": 800,
            "height": 600,
            "grid_size": 0.25,
            "render_depth": False,
            "render_instance_segmentation": False,
            "text_state_mode": "first_person",
            "platform": None,
        },
        "max_steps": 30,
        "context_management": {
            "max_trajectory_length": 20,
            "enable_long_term_summary": False,
            "short_term_history_window_size": 30,
        },
        "model": {
            "vlm": {
                "provider": "openai",
                "model_name": "gpt-4o",
                "temperature": 0.2,
                "top_p": 0.95,
                "max_tokens": 2000,
            },
        },
        "logging": {
            "stdout_verbose": True,
            "save_step_images": True,
        },
        "actions": {
            "move_small_magnitude": 0.25,
            "move_medium_magnitude": 0.5,
            "move_large_magnitude": 1,
            "move_ahead_magnitude": 0.25,
            "move_back_magnitude": 0.25,
            "move_left_magnitude": 0.25,
            "move_right_magnitude": 0.25,
            "rotate_degrees": 90,
        },
        "reward": {
            "success_reward": 10.0,
            "step_success_bonus": 0.1,
            "step_failure_penalty": -0.05,
        },
        "experiment": {
            "num_episodes": 1,
            "output_dir": "outputs",
        },
    }

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            project_root = Path(__file__).resolve().parents[2]
            config_path = project_root / "configs" / "procthor" / "config.yaml"
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self):
        import copy
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    yaml_config = yaml.safe_load(f)
                if yaml_config:
                    self.config = self._merge_configs(self.config, yaml_config)
                    print(f"✓ Configuration loaded from {self.config_path}")
            except Exception as e:
                print(f"⚠️  Failed to load config: {e}")

    def _find_task_json(self, task_name: str) -> Optional[Path]:
        """Find task.json from standard dirs; support procthor000 / procthor_000.

        Search order:
          1. <project_root>/dual_agent/task_mutil_procthor/<name>/task.json
          2. <project_root>/tasks/<name>/task.json
        """
        project_root = Path(__file__).resolve().parents[2]
        candidates = [task_name, task_name.replace("_", "")]
        if "procthor" in task_name.lower() and "_" not in task_name and len(task_name) > 7:
            candidates.append("procthor_" + task_name[7:])

        search_roots = [
            project_root / "dual_agent" / "task_mutil_procthor",
            project_root / "tasks",
            project_root / "data" / "procthor" / "tasks",
        ]
        for root in search_roots:
            for name in candidates:
                if not name:
                    continue
                p = root / name / "task.json"
                if p.exists():
                    return p
        return None

    def _load_task_from_json(self, task_json_path: Path) -> Dict[str, Any]:
        with open(task_json_path, "r", encoding="utf-8") as f:
            task_config = json.load(f)
        if "task_id" not in task_config and "task_name" not in task_config:
            task_config["task_id"] = task_json_path.parent.name
        if "instruction" in task_config and "description" not in task_config:
            task_config["description"] = task_config["instruction"]
        elif "description" in task_config and "instruction" not in task_config:
            task_config["instruction"] = task_config["description"]
        return task_config

    @staticmethod
    def _derive_golden_action_n(task_config: Dict[str, Any]) -> Optional[int]:
        """       n：   golden actions  ，     fallback """
        return derive_task_n(task_config)

    @staticmethod
    def _compute_max_steps_from_n(n: int) -> int:
        """max_steps = 10 + 2 * n."""
        return compute_max_steps_from_n(n)

    @staticmethod
    def _compute_dual_agent_max_steps_from_n(n: int) -> int:
        """Dual-agent also uses the unified per-agent cap: max_steps = 10 + 2n."""
        return compute_max_steps_from_n(n)

    def apply_task_by_name(self, task_name: str, *, dual_agent: bool = False) -> Dict[str, Any]:
        """Apply task and set max_steps from golden action count n.

        Single-agent: max_steps = 10 + 2 * n.
        Dual-agent (dual_agent=True): per-agent max_steps also uses 10 + 2 * n.

        NOTE: The value stored in task_config['max_steps'] is the PER-AGENT cap.
        Dual-agent main.py derives the global cap as 2 * per_agent_steps, so the
        two agents have independent step budgets (not a shared pool).
        """
        compute_max_steps = (
            self._compute_dual_agent_max_steps_from_n
            if dual_agent
            else self._compute_max_steps_from_n
        )
        task_json_path = self._find_task_json(task_name)
        if task_json_path:
            print(f"✓ Loading task from: {task_json_path}")
            task_config = self._load_task_from_json(task_json_path)
            task_folder_path = str(task_json_path.parent.absolute())
            self.config["task"] = {
                "name": task_config.get("task_id") or task_config.get("task_name") or task_name,
                "max_steps": self.config.get("max_steps", 30),
                "task_folder_path": task_folder_path,
                **task_config,
            }
            n = self._derive_golden_action_n(self.config["task"])
            if n is not None:
                self.config["task"]["max_steps"] = compute_max_steps(n)
            return self.config["task"]
        task_presets = self.config.get("task_presets", {})
        if task_name not in task_presets:
            available = self._get_available_json_tasks()
            raise ValueError(f"Task '{task_name}' not found. Available: {available}")
        preset = task_presets[task_name]
        self.config["task"] = {
            "name": task_name,
            "max_steps": self.config.get("max_steps", 30),
            **preset,
        }
        n = self._derive_golden_action_n(self.config["task"])
        if n is not None:
            self.config["task"]["max_steps"] = compute_max_steps(n)
        return self.config["task"]

    def _get_available_json_tasks(self) -> List[str]:
        project_root = Path(__file__).resolve().parents[2]
        names: set = set()
        for root in [
            project_root / "dual_agent" / "task_mutil_procthor",
            project_root / "tasks",
            project_root / "data" / "procthor" / "tasks",
        ]:
            if not root.exists():
                continue
            for d in root.iterdir():
                if d.is_dir() and (d / "task.json").exists():
                    names.add(d.name)
        return sorted(names)

    def get_all_task_names(self) -> List[str]:
        json_tasks = self._get_available_json_tasks()
        presets = list(self.config.get("task_presets", {}).keys())
        out = list(json_tasks)
        for p in presets:
            if p not in out:
                out.append(p)
        return out

    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        result = base.copy()
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = self._merge_configs(result[k], v)
            else:
                result[k] = v
        return result

    def get(self, key_path: str, default: Any = None) -> Any:
        val = self.config
        for key in key_path.split("."):
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                return default
        return val

    def get_section(self, section: str) -> Dict[str, Any]:
        return self.config.get(section, {})

    def get_all(self) -> Dict[str, Any]:
        return self.config

    def update(self, key_path: str, value: Any):
        keys = key_path.split(".")
        target = self.config
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value


def load_config(config_path: Optional[str] = None, **overrides) -> ConfigLoader:
    loader = ConfigLoader(config_path)
    for key, value in overrides.items():
        loader.update(key.replace("_", "."), value)
    return loader


def print_config(config: ConfigLoader, section: Optional[str] = None):
    data = config.get_section(section) if section else config.get_all()
    print(json.dumps(data, indent=2, ensure_ascii=False))
