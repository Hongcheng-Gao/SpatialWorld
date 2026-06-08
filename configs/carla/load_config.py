"""
Configuration Loading Module
Provides lightweight YAML configuration loading and management functionality
Supports loading tasks from both YAML presets and individual task.json files
"""
import json
import os
import sys
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from actions.max_steps import resolve_max_steps_from_task as _resolve_spatialworld_max_steps


def resolve_max_steps_from_task(merged_task: Dict[str, Any], yaml_default: int) -> int:
    """Resolve max_steps by the shared SpatialWorld rule: max_steps = 10 + 2n."""
    return _resolve_spatialworld_max_steps(merged_task, yaml_default)


class ConfigLoader:
    """Configuration Loader
    
    Responsible for loading YAML configuration files, providing default values and configuration access methods
    """
    
    # Default configuration (used when YAML file doesn't exist or fields are missing)
    # Note: task section does not provide default values, must be specified in configuration file
    DEFAULT_CONFIG = {
        "env": {
            "type": "ai2thor",
            "scene": "FloorPlan1",
            "width": 800,
            "height": 600,
            "grid_size": 0.25,
            "render_depth": False,
            "render_instance_segmentation": False,
            "text_state_mode": "first_person",
        },
        "max_steps": 30,  # Default maximum steps
        "context_management": {
            "max_trajectory_length": 20,
            "enable_auto_truncation": True,
        },
        "model": {
            "vlm": {
                "provider": "openai",
                "model_name": "gpt-4o",
                "temperature": 0.2,
                "max_tokens": 2000,
                "top_p": None,
                "base_url": None,
                "api_key": None,
            },
        },
        "logging": {
            "episode_log_dir": "deprecated",  # Deprecated, episode JSON is now saved in each run directory
            "stdout_verbose": True,
            "save_step_images": True,
        },
        "actions": {
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
        """Initialize configuration loader
        
        Args:
            config_path: Configuration file path, defaults to config.yaml in project root directory
        """
        if config_path is None:
            # Default to look for config.yaml in project root directory
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration file"""
        # Start with default configuration (deep copy)
        self.config = self._deep_copy_dict(self.DEFAULT_CONFIG)
        
        # If configuration file exists, load and merge
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f)
                
                if yaml_config:
                    # Recursively merge configuration
                    self.config = self._merge_configs(self.config, yaml_config)
                    print(f"✓ Configuration loaded from {self.config_path}")
                else:
                    print(f"⚠️  Configuration file {self.config_path} is empty, using default configuration")
            
            except yaml.YAMLError as e:
                print(f"⚠️  YAML parsing error: {e}")
                print("   Will use default configuration")
            
            except Exception as e:
                print(f"⚠️  Failed to load configuration file: {e}")
                print("   Will use default configuration")
        else:
            print(f"⚠️  Configuration file {self.config_path} does not exist, using default configuration")
        
        # No longer automatically apply task presets, handled by main program loop
    
    def _tasks_dir(self) -> Path:
        rel = os.environ.get("CARLA_TASKS_ROOT", "data/carla/tasks")
        tasks_dir = Path(rel)
        return tasks_dir if tasks_dir.is_absolute() else _REPO_ROOT / tasks_dir

    def _find_task_json(self, task_name: str) -> Optional[Path]:
        """Find task.json file for a given task name
        
        Searches in tasks/{task_name}/task.json
        Supports task ID format conversion (with/without underscore)
        
        Args:
            task_name: Task name/ID (e.g., "ai2thor00001" or "ai2thor_00001")
            
        Returns:
            Path to task.json if found, None otherwise
        """
        tasks_root = self._tasks_dir()

        # Method 1: Try direct path
        task_json_path = tasks_root / task_name / "task.json"
        if task_json_path.exists():
            return task_json_path
        
        # Method 2: If task_name has underscore, try without underscore
        # e.g., ai2thor_04000 -> ai2thor04000
        if '_' in task_name:
            task_name_without_underscore = task_name.replace('_', '')
            task_json_path = tasks_root / task_name_without_underscore / "task.json"
            if task_json_path.exists():
                return task_json_path
        
        # Method 3: If task_name doesn't have underscore and starts with "ai2thor",
        # try with underscore after "ai2thor"
        # e.g., ai2thor04000 -> ai2thor_04000
        if 'ai2thor' in task_name.lower() and '_' not in task_name:
            # Add underscore after "ai2thor"
            task_name_with_underscore = task_name.replace('ai2thor', 'ai2thor_', 1)
            task_json_path = tasks_root / task_name_with_underscore / "task.json"
            if task_json_path.exists():
                return task_json_path
        
        return None
    
    def _load_task_from_json(self, task_json_path: Path) -> Dict[str, Any]:
        """Load task configuration from a task.json file
        
        Args:
            task_json_path: Path to task.json file
            
        Returns:
            Task configuration dictionary
        """
        with open(task_json_path, 'r', encoding='utf-8') as f:
            task_config = json.load(f)

        task_folder = task_json_path.parent

        # Ensure required fields exist
        if "task_id" not in task_config and "task_name" not in task_config:
            task_config["task_id"] = task_folder.name

        task_config.setdefault("task_folder_path", str(task_folder.resolve()))

        image_url = task_config.get("image_url")
        if isinstance(image_url, str) and image_url:
            image_path = Path(image_url)
            if not image_path.is_absolute():
                task_config["image_url"] = str((task_folder / image_path).resolve())

        # Handle instruction/description compatibility
        # Priority: instruction > description (backward compatible)
        if "instruction" in task_config:
            # Use instruction as the primary field, also set description for backward compatibility
            if "description" not in task_config:
                task_config["description"] = task_config["instruction"]
        elif "description" in task_config:
            # Old format: only description exists, also set instruction for forward compatibility
            task_config["instruction"] = task_config["description"]
        
        return task_config
    
    def apply_task_by_name(self, task_name: str) -> Dict[str, Any]:
        """Apply task configuration by name
        
        Priority:
        1. First try to load from tasks/{task_name}/task.json
        2. Fall back to task_presets in YAML config
        
        Args:
            task_name: Task name or task ID
            
        Returns:
            Task configuration dictionary
        """
        # Priority 1: Try to load from task.json file
        task_json_path = self._find_task_json(task_name)
        if task_json_path:
            print(f"✓ Loading task from: {task_json_path}")
            task_config = self._load_task_from_json(task_json_path)
            
            # Merge with default values
            yaml_ms = self.config.get("max_steps", 30)
            self.config["task"] = {
                "name": task_config.get("task_id") or task_config.get("task_name") or task_name,
                "max_steps": yaml_ms,
                **task_config,
            }
            self.config["task"]["max_steps"] = resolve_max_steps_from_task(
                self.config["task"], yaml_ms
            )

            return self.config["task"]
        
        # Priority 2: Fall back to task_presets
        task_presets = self.config.get("task_presets", {})
        
        # Check if task preset exists
        if task_name not in task_presets:
            available_presets = list(task_presets.keys())
            available_json_tasks = self._get_available_json_tasks()
            
            error_msg = f"❌ Configuration error: Task '{task_name}' not found\n"
            if available_presets:
                error_msg += f"   Available task presets: {', '.join(available_presets)}\n"
            if available_json_tasks:
                error_msg += f"   Available task.json tasks: {', '.join(available_json_tasks)}\n"
            error_msg += "   Please check if the task name is correct."
            
            raise ValueError(error_msg)
        
        # Build task configuration from preset
        preset = task_presets[task_name]
        
        yaml_ms = self.config.get("max_steps", 30)
        self.config["task"] = {
            "name": task_name,
            "max_steps": yaml_ms,
            **preset,
        }
        self.config["task"]["max_steps"] = resolve_max_steps_from_task(
            self.config["task"], yaml_ms
        )

        return self.config["task"]
    
    def _get_available_json_tasks(self) -> List[str]:
        """Get list of available task.json tasks
        
        Returns:
            List of task IDs that have task.json files
        """
        tasks_dir = self._tasks_dir()
        
        if not tasks_dir.exists():
            return []
        
        available_tasks = []
        for task_dir in tasks_dir.iterdir():
            if task_dir.is_dir():
                task_json = task_dir / "task.json"
                if task_json.exists():
                    available_tasks.append(task_dir.name)
        
        return sorted(available_tasks)
    
    def get_all_task_names(self) -> List[str]:
        """Get list of all available task names
        
        Combines tasks from both task_presets and task.json files
        
        Returns:
            List of task names (task.json tasks first, then presets)
        """
        # Get task.json tasks
        json_tasks = self._get_available_json_tasks()
        
        # Get preset tasks
        preset_tasks = list(self.config.get("task_presets", {}).keys())
        
        # Combine (json tasks first, then presets not in json)
        all_tasks = json_tasks.copy()
        for preset in preset_tasks:
            if preset not in all_tasks:
                all_tasks.append(preset)
        
        return all_tasks
    
    def _deep_copy_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Deep copy dictionary"""
        import copy
        return copy.deepcopy(d)
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two configuration dictionaries
        
        Args:
            base: Base configuration (contains default values)
            override: Override configuration (from YAML file)
        
        Returns:
            Merged configuration
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self._merge_configs(result[key], value)
            else:
                # Direct override
                result[key] = value
        
        return result
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value (supports dot-separated paths)
        
        Args:
            key_path: Configuration key path, e.g., "model.vlm.temperature"
            default: Default value
        
        Returns:
            Configuration value
        
        Example:
            >>> config = ConfigLoader()
            >>> config.get("model.vlm.temperature")
            0.2
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get a top-level section of configuration
        
        Args:
            section: Section name, e.g., "env", "task", "model"
        
        Returns:
            Configuration dictionary for that section
        """
        return self.config.get(section, {})
    
    def get_all(self) -> Dict[str, Any]:
        """Get complete configuration"""
        return self.config
    
    def update(self, key_path: str, value: Any):
        """Update configuration value
        
        Args:
            key_path: Configuration key path
            value: New value
        """
        keys = key_path.split('.')
        target = self.config
        
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        target[keys[-1]] = value
    
    def build_success_predicate(self) -> callable:
        """Build success condition predicate function based on success_condition configuration
        
        Returns:
            lambda function that receives object metadata and returns whether condition is met
        """
        success_condition = self.config["task"]["success_condition"]
        condition_type = success_condition["type"]
        
        if condition_type == "object_state":
            field = success_condition["field"]
            target_value = success_condition["value"]
            
            return lambda obj: obj.get(field, False) == target_value
        
        else:
            # Default: never satisfied (avoid misjudgment)
            print(f"⚠️  Unsupported success_condition type: {condition_type}")
            return lambda obj: False


def load_config(config_path: Optional[str] = None, **overrides) -> ConfigLoader:
    """Load configuration file (convenience function)
    
    Args:
        config_path: Configuration file path
        **overrides: Command line override parameters
    
    Returns:
        ConfigLoader instance
    
    Example:
        >>> config = load_config("config.yaml", scene="FloorPlan2", max_steps=50)
    """
    loader = ConfigLoader(config_path)
    
    # Apply command line overrides
    for key, value in overrides.items():
        # Convert underscore to dot (command line parameter convention)
        key_path = key.replace('_', '.')
        loader.update(key_path, value)
    
    return loader


def print_config(config: ConfigLoader, section: Optional[str] = None):
    """Print configuration information
    
    Args:
        config: ConfigLoader instance
        section: Section to print, None means print all
    """
    import json
    
    if section:
        data = config.get_section(section)
        print(f"\nConfiguration Section: {section}")
    else:
        data = config.get_all()
        print("\nComplete Configuration:")
    
    print("=" * 60)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print("=" * 60)


# ==================== Usage Examples ====================

if __name__ == "__main__":
    # Example 1: Load default configuration
    config = load_config()
    print_config(config)
    
    # Example 2: Get specific configuration values
    print(f"\nVLM Model: {config.get('model.vlm.model_name')}")
    print(f"Scene: {config.get('env.scene')}")
    print(f"Max Steps: {config.get('task.max_steps')}")
    
    # Example 3: Get configuration section
    task_config = config.get_section('task')
    print(f"\nTask Configuration: {task_config}")
    
    # Example 4: Build success condition predicate
    predicate = config.build_success_predicate()
    test_obj_open = {"isOpen": True}
    test_obj_closed = {"isOpen": False}
    print(f"\nObject (isOpen=True) meets condition: {predicate(test_obj_open)}")
    print(f"Object (isOpen=False) meets condition: {predicate(test_obj_closed)}")
