"""
ProcTHOR Environment Wrapper
   spatial-planning    AI2-THOR       
"""

import os
import sys
import json
import gzip
import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable, List
from datetime import datetime
from PIL import Image
import numpy as np

# prior      GitHub API；     403/404     ~/.git-credentials 
#       .git-credentials（   prior     token   API   401） 
#        API    403，prior       ~/.prior           GITHUB_TOKEN 

#    Git         ~/.git-credentials    FileNotFoundError
# （prior/huggingface     git，       credential.helper=store           ）
if not os.environ.get("GIT_CONFIG_GLOBAL"):
    try:
        import tempfile
        _fd, _path = tempfile.mkstemp(suffix=".gitconfig", prefix="procthor_")
        os.close(_fd)
        with open(_path, "w") as _f:
            _f.write("[credential]\n\thelper = \n")
        os.environ["GIT_CONFIG_GLOBAL"] = _path
    except Exception:
        pass

import prior
from ai2thor.controller import Controller
try:
    from ai2thor.platform import CloudRendering
except Exception:
    CloudRendering = None

#          ，     core   
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm.schemas import EnvObservation, EnvAction
from envs.base import BaseEnv


# ============================================================================
#      ：    ->           
#          ，      PickupObject(Apple)  ，
#         ["Apple", "AppleSliced"]         
# ============================================================================
SEMANTIC_OBJECT_MAPPING = {
    #      
    "Apple": ["Apple", "AppleSliced"],
    "Bread": ["Bread", "BreadSliced", "BreadToasted"],
    "Tomato": ["Tomato", "TomatoSliced"],
    "Lettuce": ["Lettuce", "LettuceSliced"],
    "Potato": ["Potato", "PotatoSliced", "PotatoCooked"],
    "Egg": ["Egg", "EggSliced", "EggBroken", "EggCooked"],
    #      
    "Bottle": ["Bottle", "BottleBroken"],
    "Cup": ["Cup", "CupBroken"],
    "Mug": ["Mug", "MugBroken"],
    "Plate": ["Plate", "PlateBroken"],
    "Vase": ["Vase", "VaseBroken"],
    "WineBottle": ["WineBottle", "WineBottleBroken"],
    "Window": ["Window", "WindowBroken"],
    "Statue": ["Statue", "StatueBroken"],
    #      
    "PaperTowelRoll": ["PaperTowelRoll", "PaperTowel"],
}


class ProcTHOREnvWrapper(BaseEnv):
    """ProcTHOR      
    
       AI2-THOR Controller，   ProcTHOR-10K    
       spatial-planning/envs/ai2thor      
    """
    
    def __init__(
        self,
        scene: Optional[Any] = None,  #           
        scene_index: int = 0,  #    scene   None，     
        dataset_name: str = "procthor-10k",
        grid_size: float = 0.25,
        render_depth_image: bool = False,
        render_instance_segmentation: bool = False,
        width: int = 800,
        height: int = 600,
        field_of_view: int = 60,
        visibility_distance: float = 1.0,
        output_dir: str = "outputs",
        config: Optional[Dict[str, Any]] = None,
        headless: bool = False,
    ):
        """    ProcTHOR      
        
        Args:
            scene: ProcTHOR     （house），    None        
            scene_index:     （  scene   None    ）
            dataset_name:      ，   "procthor-10k"
            grid_size:       （ ）
            render_depth_image:        
            render_instance_segmentation:          
            width:     （  ）
            height:     （  ）
            field_of_view:       （ ）
            visibility_distance:          （ ）
            output_dir:     
            config:     
            headless:       
        """
        #        
        super().__init__(config)
        
        #       
        self.config = config or {}
        
        #          
        env_config = self.config.get("env", {})
        self.scene_index = scene_index
        self.dataset_name = dataset_name
        self.grid_size = env_config.get("grid_size", grid_size)
        self.render_depth_image = env_config.get("render_depth", render_depth_image)
        self.render_instance_segmentation = env_config.get("render_instance_segmentation", render_instance_segmentation)
        self.width = env_config.get("width", width)
        self.height = env_config.get("height", height)
        self.field_of_view = env_config.get("field_of_view", field_of_view)
        self.visibility_distance = env_config.get("visibility_distance", visibility_distance)
        self.text_state_mode = env_config.get("text_state_mode", "first_person")
        self.agent_count = int(env_config.get("agent_count", 1))
        if self.agent_count < 1:
            self.agent_count = 1
        self.headless = bool(env_config.get("headless", headless))
        # AI2-THOR backend timeout settings (seconds)
        self.controller_timeout = float(env_config.get("server_timeout", env_config.get("timeout", 100.0)))
        self.controller_start_timeout = float(env_config.get("server_start_timeout", 300.0))
        # xvfb/physical X display (e.g. ":99"), allow explicit config first, then env DISPLAY
        self.x_display = self._normalize_x_display(
            env_config.get("x_display", os.environ.get("DISPLAY"))
        )
        # headless    ，      DISPLAY          X socket（   xvfb :1），
        #     X display   （OpenGL），          CloudRendering（Vulkan） 
        if bool(env_config.get("headless", headless)) and not self.x_display:
            auto_disp = self._auto_detect_x_display()
            if auto_disp:
                self.x_display = auto_disp
                print(f"  • Auto-detected X display: {auto_disp} (will skip CloudRendering fallback)")
        if self.text_state_mode not in {"first_person", "omniscient"}:
            print(f"⚠️  Unknown text_state_mode: {self.text_state_mode}, falling back to first_person")
            self.text_state_mode = "first_person"
        platform_setting = env_config.get("platform")
        if (
            isinstance(platform_setting, str)
            and platform_setting.lower() == "cloudrendering"
            and CloudRendering is not None
        ):
            self.controller_platform = CloudRendering
        else:
            self.controller_platform = platform_setting
        #     ：      /   DISPLAY（xvfb   ），     CloudRendering
        if self.headless and self.controller_platform is None:
            if self.x_display:
                print(f"  • Headless mode with X display: {self.x_display}")
            elif CloudRendering is not None:
                self.controller_platform = CloudRendering
                print("  • Headless mode fallback to CloudRendering (no DISPLAY detected)")
        
        #     （     ，  AI2-THOR   ：  /  /    ）
        actions_config = self.config.get("actions", {})
        self.move_small_magnitude = actions_config.get("move_small_magnitude", 0.25)
        self.move_medium_magnitude = actions_config.get("move_medium_magnitude", 0.5)
        self.move_large_magnitude = actions_config.get("move_large_magnitude", 1.0)
        self.move_ahead_magnitude = actions_config.get("move_ahead_magnitude", self.move_small_magnitude)
        self.move_back_magnitude = actions_config.get("move_back_magnitude", self.move_small_magnitude)
        self.move_left_magnitude = actions_config.get("move_left_magnitude", self.move_small_magnitude)
        self.move_right_magnitude = actions_config.get("move_right_magnitude", self.move_small_magnitude)
        self.rotate_degrees = actions_config.get("rotate_degrees", 90)
        
        #     （     ）
        reward_config = self.config.get("reward", {})
        self.success_reward = reward_config.get("success_reward", 10.0)
        self.step_success_bonus = reward_config.get("step_success_bonus", 0.1)
        self.step_failure_penalty = reward_config.get("step_failure_penalty", -0.05)
        
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        #       （         ）
        if scene is None:
            print(f"Loading ProcTHOR dataset: {dataset_name}")
            online_error: Optional[Exception] = None
            offline_errors: List[str] = []
            self.dataset = None

            #       jsonl.gz   ~/.prior          ，   prior
            #   GitHub API          ~/.git-credentials 
            local_scene, local_source = self._load_scene_from_local_dataset(
                dataset_name, scene_index
            )
            if local_scene is not None:
                self.scene = local_scene
                print(f"✓ Loaded local ProcTHOR scene {scene_index}: {local_source}")
            else:
                cached_scene, cached_source = self._load_scene_from_prior_cache(
                    dataset_name, scene_index
                )
                if cached_scene is not None:
                    self.scene = cached_scene
                    print(
                        f"✓ Loaded cached prior ProcTHOR scene {scene_index}: {cached_source}"
                    )
                else:
                    try:
                        self.dataset = prior.load_dataset(dataset_name)
                    except Exception as e1:
                        online_error = e1
                        for revision in [
                            "main",
                            "ab3cacd0fc17754d4c080a3fd50b18395fae8647",
                        ]:
                            try:
                                self.dataset = prior.load_dataset(
                                    dataset_name, revision=revision, offline=True
                                )
                                print(f"✓ Loaded prior dataset from offline cache: {revision}")
                                break
                            except Exception as e2:
                                offline_errors.append(f"{revision}: {e2}")

                        if self.dataset is None:
                            offline_detail = "\n".join(offline_errors) or " "
                            raise RuntimeError(
                                "Failed to load ProcTHOR dataset:\n"
                                "  1. Set a local dataset directory: "
                                "export PROCTHOR_DATASET_DIR=/path/to/procthor-10k\n"
                                "  2. Or provide a GitHub token: "
                                "export GITHUB_TOKEN=your_github_token\n"
                                "  3. Or store credentials in ~/.git-credentials: "
                                "https://<token>@github.com\n"
                                "  4. Or use the prior offline cache under ~/.prior\n"
                                "     Common local paths: ./datasets/procthor-10k or "
                                "./procthor-10k\n"
                                f"Online error: {online_error}\n"
                                f"Offline errors:\n{offline_detail}"
                            ) from e1
            if self.dataset is not None:
                self.scene = self.dataset["train"][scene_index]
                print(f"Loaded ProcTHOR dataset scene: {scene_index}")
        else:
            self.scene = scene
            self.dataset = None
        
        #     Controller（    reset    ）
        self.controller = None
        self.task_description = ""
        
        #     （     ）
        task_config = self.config.get("task", {})
        self.target_object_types: List[str] = task_config.get("target_object_types", [])
        self.success_condition: Optional[Dict[str, Any]] = task_config.get("success_condition", None)
        self.success_predicate: Optional[Callable[[dict], bool]] = None  # Will be built when needed
        self.target_description: str = task_config.get("target_description", "")
        
        #           ，        
        if self.success_condition:
            self.success_predicate = self._build_success_predicate_from_config()
        
        print("✓ ProcTHOR environment initialized")

    @staticmethod
    def _normalize_x_display(raw_display: Optional[Any]) -> Optional[str]:
        """Normalize x_display input to AI2-THOR compatible format."""
        if raw_display is None:
            return None
        d = str(raw_display).strip()
        if not d:
            return None
        if d.isdigit():
            return f":{d}"
        return d

    @staticmethod
    def _auto_detect_x_display() -> Optional[str]:
        """   /tmp/.X11-unix/X*          X display 

          `xdpyinfo`       ；  xdpyinfo          socket        
                   None 
        """
        try:
            sock_dir = Path("/tmp/.X11-unix")
            if not sock_dir.is_dir():
                return None
            candidates: list[str] = []
            for sock in sorted(sock_dir.iterdir()):
                name = sock.name
                if not name.startswith("X"):
                    continue
                suffix = name[1:]
                if suffix.isdigit():
                    candidates.append(f":{suffix}")
            if not candidates:
                return None
            xdpyinfo = shutil.which("xdpyinfo")
            if xdpyinfo is None:
                return candidates[0]
            for disp in candidates:
                try:
                    rc = subprocess.run(
                        [xdpyinfo],
                        env={**os.environ, "DISPLAY": disp},
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=3,
                    ).returncode
                    if rc == 0:
                        return disp
                except Exception:
                    continue
            return None
        except Exception:
            return None

    def _resolve_local_dataset_dir(self, dataset_name: str) -> Optional[Path]:
        """     ProcTHOR       

           ：
        1) config.env.dataset_dir
        2)      PROCTHOR_DATASET_DIR
        3) <project_root>/datasets/<dataset_name>
        4) <project_root>/<dataset_name>
        5) <cwd>/datasets/<dataset_name>
        6) <cwd>/<dataset_name>
        """
        env_cfg = self.config.get("env", {}) if isinstance(self.config, dict) else {}
        candidates: List[Path] = []

        cfg_dir = env_cfg.get("dataset_dir")
        if cfg_dir:
            candidates.append(Path(cfg_dir).expanduser())

        env_dir = os.environ.get("PROCTHOR_DATASET_DIR")
        if env_dir:
            candidates.append(Path(env_dir).expanduser())

        project_root = Path(__file__).resolve().parents[1]
        candidates.append(project_root / "datasets" / dataset_name)
        candidates.append(project_root / dataset_name)
        candidates.append(Path.cwd() / "datasets" / dataset_name)
        candidates.append(Path.cwd() / dataset_name)

        for c in candidates:
            try:
                c = c.resolve()
            except Exception:
                continue
            if c.exists() and c.is_dir() and (c / "train.jsonl.gz").exists():
                return c
        return None

    def _load_scene_from_local_dataset(
        self, dataset_name: str, scene_index: int
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """    jsonl.gz   scene_index       （   train split） """
        dataset_dir = self._resolve_local_dataset_dir(dataset_name)
        if dataset_dir is None:
            return None, None

        env_cfg = self.config.get("env", {}) if isinstance(self.config, dict) else {}
        split = env_cfg.get("dataset_split", "train")
        data_file = dataset_dir / f"{split}.jsonl.gz"
        return self._load_scene_from_jsonl_file(data_file, scene_index)

    def _load_scene_from_prior_cache(
        self, dataset_name: str, scene_index: int
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """  prior   ~/.prior           ，     API/token    """
        home = Path.home()
        project_dir = home / ".prior" / "datasets" / "allenai" / dataset_name
        if not project_dir.is_dir():
            return None, None

        candidate_dirs: List[Path] = []
        cache_file = project_dir / "cache"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                for key in ("main", "HEAD"):
                    sha = cache.get(key)
                    if sha:
                        candidate_dirs.append(project_dir / sha)
                for sha in cache.values():
                    if sha:
                        candidate_dirs.append(project_dir / sha)
            except Exception:
                pass

        for child in sorted(project_dir.iterdir()):
            if child.is_dir():
                candidate_dirs.append(child)

        seen = set()
        for dataset_dir in candidate_dirs:
            try:
                dataset_dir = dataset_dir.resolve()
            except Exception:
                continue
            if dataset_dir in seen:
                continue
            seen.add(dataset_dir)
            data_file = dataset_dir / "train.jsonl.gz"
            if not data_file.exists():
                continue
            scene, source = self._load_scene_from_jsonl_file(data_file, scene_index)
            if scene is not None:
                return scene, source

        return None, None

    @staticmethod
    def _load_scene_from_jsonl_file(
        data_file: Path, scene_index: int
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """  jsonl.gz           ProcTHOR    """
        if not data_file.exists():
            return None, str(data_file)

        try:
            with gzip.open(data_file, "rt", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if idx == scene_index:
                        return json.loads(line), str(data_file)
        except Exception:
            return None, str(data_file)
        return None, str(data_file)
    
    def reset(self, task_description: str = "", scene: Optional[Any] = None) -> EnvObservation:
        """    
        
        Args:
            task_description:     
            scene:        （      ）
            
        Returns:
            EnvObservation:      
        """
        #         ，   
        if scene is not None:
            self.scene = scene
        
        self.task_description = task_description
        self.step_counter = 0  #    step_counter     BaseEnv
        
        #          ，      （    ）
        if not self.target_object_types:
            print("⚠️  Warning: Task parameters not configured, using default task (open Fridge)")
            self.configure_task(
                target_object_types=["Fridge"],
                success_predicate=lambda obj: obj.get("isOpen", False),
                target_description="Find and open any Fridge",
            )
        
        #       Controller
        if self.controller is not None:
            self.controller.stop()
        
        print("Initializing ProcTHOR scene...")
        # AI2-THOR   Linux + xvfb    headless=True      ChangeResolution 
        #       x_display  ，       （headless=False）    
        runtime_headless = self.headless and not bool(self.x_display)
        # ai2thor Controller.step()：  headless=True     action["renderImage"]=False，   event.frame   
        # None（  RGB，VLM     ） CloudRendering    GPU     ，   headless=False      
        if CloudRendering is not None and self.controller_platform is CloudRendering:
            runtime_headless = False
            print("  • CloudRendering: Controller headless=False (required for RGB / renderImage)")
        controller_kwargs = {
            "scene": self.scene,
            "headless": runtime_headless,
            "gridSize": self.grid_size,
            "width": self.width,
            "height": self.height,
            "fieldOfView": self.field_of_view,
            "renderDepthImage": self.render_depth_image,
            "renderInstanceSegmentation": self.render_instance_segmentation,
            "visibilityDistance": self.visibility_distance,
            "server_timeout": self.controller_timeout,
            "server_start_timeout": self.controller_start_timeout,
        }
        if self.x_display:
            controller_kwargs["x_display"] = self.x_display
            print(f"  • Using x_display: {self.x_display}")
            if self.headless and not runtime_headless:
                print("  • headless=True with x_display detected, using headless=False for stability")
        if self.controller_platform is not None:
            controller_kwargs["platform"] = self.controller_platform
            platform_name = (
                "CloudRendering" if (CloudRendering is not None and self.controller_platform is CloudRendering) else str(self.controller_platform)
            )
            print(f"  • Using controller platform: {platform_name}")
        # ProcTHOR + AI2-THOR v5         ：
        #     Controller(scene=<procthor_house>, agentCount=2)   ，AI2-THOR      
        # Reset(Procedural) → Initialize(agentCount=2) → CreateHouse(house)    CreateHouse
        #    agent      1，    agentId=1             （fifo   ） 
        #       ：      Controller（   scene），    CreateHouse + Initialize(agentCount=N) 
        deferred_procthor_load = bool(self.agent_count > 1)
        if deferred_procthor_load:
            controller_kwargs.pop("scene", None)
            print(
                f"  • Multi-agent embodied mode: deferred CreateHouse + "
                f"Initialize(agentCount={self.agent_count})"
            )

        try:
            self.controller = Controller(**controller_kwargs)
        except TypeError as e:
            # Backward compatibility: some AI2-THOR builds use timeout instead of server_timeout
            if "unexpected keyword argument 'server_timeout'" in str(e):
                controller_kwargs.pop("server_timeout", None)
                controller_kwargs["timeout"] = self.controller_timeout
            if "unexpected keyword argument 'server_start_timeout'" in str(e):
                controller_kwargs.pop("server_start_timeout", None)
            self.controller = Controller(**controller_kwargs)

        if deferred_procthor_load:
            # CreateHouse        lastActionSuccess=False（NullReferenceException），  house
            #       Unity    ；   Initialize(agentCount=N)     respawn       agents 
            try:
                ch_ev = self.controller.step(action="CreateHouse", house=self.scene)
                ch_ok = bool(ch_ev.metadata.get("lastActionSuccess", True))
                print(
                    f"  • CreateHouse ok={ch_ok} "
                    "(ProcTHOR may report lastActionSuccess=False even when the house loads)"
                )
            except Exception as e:
                print(f"⚠️  CreateHouse failed: {e}")
            try:
                init_ev = self.controller.step(
                    action="Initialize",
                    agentCount=self.agent_count,
                    gridSize=self.grid_size,
                    visibilityDistance=self.visibility_distance,
                    renderDepthImage=self.render_depth_image,
                    renderInstanceSegmentation=self.render_instance_segmentation,
                    fieldOfView=self.field_of_view,
                )
                ok = bool(init_ev.metadata.get("lastActionSuccess", True))
                n_evs = len(init_ev.events) if hasattr(init_ev, "events") else 1
                print(
                    f"✓ Re-Initialize(agentCount={self.agent_count}): "
                    f"ok={ok}, events={n_evs}"
                )
            except Exception as e:
                print(f"⚠️  Re-Initialize(agentCount={self.agent_count}) failed: {e}")

        print("✓ Controller initialized")
        
        #          （cameraHorizon ≈ 0）
        # AI2-THOR           ，         
        #         ，     0（   ）
        current_horizon = self.controller.last_event.metadata.get("agent", {}).get("cameraHorizon", 0)
        target_horizon = 0  #     ：  
        tolerance = 1.0  #     ：1 
        
        if abs(current_horizon - target_horizon) > tolerance:
            #            （   LookUp/LookDown   30  ）
            max_steps = 10  #       
            step_count = 0
            
            while step_count < max_steps:
                current_horizon = self.controller.last_event.metadata.get("agent", {}).get("cameraHorizon", 0)
                
                #         ，    
                if abs(current_horizon - target_horizon) <= tolerance:
                    break
                
                #             
                if current_horizon > target_horizon:
                    #      ，    
                    self.controller.step(action="LookUp")
                elif current_horizon < target_horizon:
                    #      ，    
                    self.controller.step(action="LookDown")
                
                step_count += 1
            
            #       
            final_horizon = self.controller.last_event.metadata.get("agent", {}).get("cameraHorizon", 0)
            if abs(final_horizon - target_horizon) <= tolerance:
                print(f"✓ Camera horizon reset (cameraHorizon: {final_horizon:.1f}°)")
            else:
                print(
                    f"⚠️  Camera horizon is {final_horizon:.1f}° "
                    f"(target: {target_horizon}°)"
                )
        else:
            #      ，     
            print(f"✓ Camera horizon already level (cameraHorizon: {current_horizon:.1f}°)")
        
        #      init_actions（       init.json），        
        init_actions = self.config.get("init_actions") or []
        if init_actions:
            print(f"Executing init actions ({len(init_actions)} actions)...")
            for i, action_dict in enumerate(init_actions, 1):
                if action_dict.get("action_type") == "task_completion":
                    print("Skipping DONE in init actions")
                    break
                try:
                    self.step_with_action_dict(action_dict)
                except Exception as e:
                    print(f"Init action {i} failed: {e}")
            print("✓ Init actions complete")
        
        #        
        return self._get_current_observation()
    
    def _unwrap_agent_event(
        self, event: Any, thor_agent_id: Optional[int] = None
    ) -> Any:
        """Multi-agent:   ``MultiAgentEvent``     per-agent ``Event``（  ``.frame``） 

           spatial-planning/envs/ai2thor/wrapper.py — ``agent_count==1``  
        ``controller.reset/step``    ``Event``；      ``MultiAgentEvent``（  ``.frame``），
            ``event.events[agent_id]`` 
        """
        if event is None:
            return None
        if getattr(event, "frame", None) is not None:
            return event
        evs = getattr(event, "events", None)
        if isinstance(evs, list) and evs:
            if thor_agent_id is not None:
                i = int(thor_agent_id)
                if 0 <= i < len(evs):
                    return evs[i]
            return evs[0]
        return event

    @staticmethod
    def _agent_position_yaw_from_metadata(
        metadata: dict, agent_index: int
    ) -> Optional[Tuple[float, float, float, float]]:
        """Read (x, y, z, yaw_deg) for an embodied agent from THOR metadata."""
        agents = metadata.get("agents") if isinstance(metadata, dict) else None
        if isinstance(agents, list) and len(agents) > agent_index:
            a = agents[agent_index] or {}
            p = a.get("position") or {}
            r = a.get("rotation") or {}
            return (
                float(p.get("x", 0.0)),
                float(p.get("y", 0.0)),
                float(p.get("z", 0.0)),
                float(r.get("y", 0.0)),
            )
        if agent_index == 0 and metadata.get("agent"):
            a = metadata["agent"]
            p = a.get("position") or {}
            r = a.get("rotation") or {}
            return (
                float(p.get("x", 0.0)),
                float(p.get("y", 0.0)),
                float(p.get("z", 0.0)),
                float(r.get("y", 0.0)),
            )
        return None

    def _probe_agent_can_move_ahead_and_back(self, thor_agent_id: int) -> bool:
        """True if agent can MoveAhead then MoveBack (navigable, not stuck)."""
        m = self.move_small_magnitude
        try:
            ev = self.controller.step(
                action="MoveAhead",
                moveMagnitude=m,
                agentId=int(thor_agent_id),
            )
            if not ev.metadata.get("lastActionSuccess"):
                return False
            ev2 = self.controller.step(
                action="MoveBack",
                moveMagnitude=m,
                agentId=int(thor_agent_id),
            )
            return bool(ev2.metadata.get("lastActionSuccess"))
        except Exception as e:
            print(f"  • probe MoveAhead/Back failed for agentId={thor_agent_id}: {e}")
            return False

    def relocate_second_agent_near_agent1(self) -> bool:
        """Place agentId=1 adjacent to agentId=0 (forward/back/left/right in agent0's frame).

           spatial-planning/envs/ai2thor/wrapper.py：
          - ``config['dual_agent']['relocate_agent2_near_agent1']`` (default True)
          - ``config['dual_agent']['second_agent_spawn_offset_m']`` (default 0.75)

        Uses TeleportFull for placement, then MoveAhead/MoveBack probe to reject stuck poses.
        """
        if self.agent_count < 2:
            return True

        da = (self.config.get("dual_agent") or {}) if isinstance(self.config, dict) else {}
        if da.get("relocate_agent2_near_agent1") is False:
            print("  ℹ️  Skipping agent2 relocation (relocate_agent2_near_agent1=false)")
            return True

        offset_m = float(da.get("second_agent_spawn_offset_m", 0.75))
        if offset_m <= 0:
            offset_m = 0.75

        ev0 = self.controller.step(action="Pass", agentId=0)
        pose0 = self._agent_position_yaw_from_metadata(ev0.metadata, 0)
        if pose0 is None:
            print("⚠️  relocate_second_agent_near_agent1: could not read agent 1 pose")
            return False

        x0, y0, z0, yaw_deg = pose0
        yaw_rad = math.radians(yaw_deg)
        # Horizontal forward / right in XZ from body yaw (THOR convention)
        fx = math.sin(yaw_rad)
        fz = math.cos(yaw_rad)
        rx = math.cos(yaw_rad)
        rz = -math.sin(yaw_rad)

        direction_offsets: List[Tuple[str, float, float]] = [
            ("forward", fx * offset_m, fz * offset_m),
            ("back", -fx * offset_m, -fz * offset_m),
            ("left", -rx * offset_m, -rz * offset_m),
            ("right", rx * offset_m, rz * offset_m),
        ]

        print(
            f"\n{'=' * 60}\n"
            f"Dual-agent spawn: placing agent 2 near agent 1 "
            f"(offset={offset_m:.2f}m, order=forward/back/left/right)\n"
            f"{'=' * 60}"
        )

        rotation = {"x": 0, "y": yaw_deg, "z": 0}

        for label, dx, dz in direction_offsets:
            tx = x0 + dx
            tz = z0 + dz
            ty = y0
            try:
                ev = self.controller.step(
                    action="TeleportFull",
                    x=tx,
                    y=ty,
                    z=tz,
                    rotation=rotation,
                    horizon=0,
                    standing=True,
                    forceAction=True,
                    agentId=1,
                )
            except Exception as e:
                print(f"  • {label}: TeleportFull raised — {e}")
                continue
            if not ev.metadata.get("lastActionSuccess"):
                err = ev.metadata.get("errorMessage", "")
                print(f"  • {label}: TeleportFull failed — {err or 'unknown'}")
                continue

            if self._probe_agent_can_move_ahead_and_back(1):
                print(
                    f"  ✓ Agent 2 placed to the {label} of agent 1 "
                    f"(probe MoveAhead/MoveBack OK)"
                )
                return True

            print(f"  • {label}: teleported but navigation probe failed, trying next")

        print(
            "⚠️  Could not place agent 2 in any of front/back/left/right; "
            "leaving default spawn for agentId=1"
        )
        return False

    def get_observation_for_agent(self, thor_agent_id: int) -> EnvObservation:
        """Render current world state from a specific embodied agent's camera (Pass, no step++).

           spatial-planning/envs/ai2thor/wrapper.py.get_observation_for_agent 
        """
        step_kwargs: Dict[str, Any] = {"action": "Pass"}
        if self.agent_count > 1:
            step_kwargs["agentId"] = int(thor_agent_id)
        event = self.controller.step(**step_kwargs)
        ev = self._unwrap_agent_event(event, thor_agent_id=int(thor_agent_id))
        image_path = self._save_frame(
            ev.frame,
            prefix=f"step_{self.step_counter}",
            thor_agent_id=int(thor_agent_id) if self.agent_count > 1 else None,
        )
        text_state = self._generate_text_state(event.metadata)
        done = self._check_done(event.metadata) if hasattr(self, "_check_done") else False
        return EnvObservation(
            image_path=image_path,
            text_state=text_state,
            reward=0.0,
            done=bool(done),
            metadata=event.metadata,
        )

    def _build_success_predicate_from_config(self) -> Callable[[dict], bool]:
        """           
        
           spatial-planning/envs/ai2thor/wrapper.py
        
        Returns:
                    
        """
        if not self.success_condition:
            return lambda obj: False
        
        condition_type = self.success_condition.get("type", "object_state")
        
        if condition_type == "object_state":
            field = self.success_condition.get("field", "isOpen")
            target_value = self.success_condition.get("value", True)
            return lambda obj: obj.get(field, False) == target_value
        
        elif condition_type == "object_in_receptacle":
            receptacle_type = self.success_condition.get("receptacle_type", "Plate")
            expected_value = self.success_condition.get("value", True)
            
            def check_in_receptacle(obj):
                parent_receptacles = obj.get("parentReceptacles", [])
                if not parent_receptacles:
                    in_receptacle = False
                else:
                    in_receptacle = False
                    for parent_id in parent_receptacles:
                        parent_type = parent_id.split("|")[0] if "|" in parent_id else parent_id
                        if parent_type == receptacle_type:
                            in_receptacle = True
                            break
                return in_receptacle == expected_value
            
            return check_in_receptacle
        
        elif condition_type == "object_in_hand":
            return lambda obj: obj.get("isPickedUp", False)
        
        else:
            print(f"⚠️  Unsupported success_condition type: {condition_type}")
            return lambda obj: False
    
    def configure_task(
        self,
        target_object_types: List[str],
        success_predicate: Callable[[dict], bool],
        target_description: str,
    ):
        """      （        ）
        
           spatial-planning/envs/ai2thor/wrapper.py
        
        Args:
            target_object_types:         （  ["Fridge"]   ["Microwave", "Cabinet"]）
            success_predicate:       ，         ，   bool
            target_description:          （    text_state    ）
        """
        self.target_object_types = target_object_types
        self.success_predicate = success_predicate
        self.target_description = target_description
        print(f"✓ Task configuration complete: {target_description}")
    
    def step_with_action_dict(
        self,
        action_dict: Dict[str, Any],
        thor_agent_id: Optional[int] = None,
    ) -> Tuple[EnvObservation, Optional[str]]:
        """    （  ：  ``thor_agent_id``        ）

        Args:
            action_dict:     ，   action_type, action_name  
            thor_agent_id:             （AI2-THOR ``agentId``，0    agent1） 

        Returns:
            Tuple[EnvObservation, Optional[str]]: (   ,     )
        """
        if self.controller is None:
            raise RuntimeError("Environment is not initialized; call reset() first")
        
        self.step_counter += 1  #    step_counter     BaseEnv
        
        action_type = action_dict.get("action_type")
        action_name = action_dict.get("action_name")
        object_type = action_dict.get("object_type")
        magnitude = action_dict.get("magnitude")
        degrees = action_dict.get("degrees")
        fill_liquid = action_dict.get("fillLiquid")
        
        print(f"\n--- Step {self.step_counter} ---")
        print(f"🎬 Action: {action_name}" + (f"({object_type})" if object_type else ""))
        
        error_message = None
        thor_action = None
        
        #     ：      ，      
        if action_type == "navigation":
            mag = self._resolve_move_magnitude(
                magnitude=action_dict.get("magnitude"),
                granularity=action_dict.get("granularity"),
            )
            deg = action_dict.get("degrees")
            thor_action = self._convert_navigation_action(action_name, magnitude=mag, degrees=deg)
        elif action_type == "interaction":
            thor_action, error_message = self._convert_interaction_action(action_name, object_type)
            if thor_action and action_name == "FillObjectWithLiquid" and fill_liquid:
                thor_action["fillLiquid"] = fill_liquid
                print(f"  💧 Liquid type: {fill_liquid}")
        elif action_type == "pass":
            thor_action = {"action": "Pass"}
            print("  ⏭️  Pass (no-op)")
        elif action_type == "task_completion":
            thor_action = {"action": "Pass"}
            print(f"🏁 Task completion action: {action_name}")
        else:
            error_message = f"Unknown action type: {action_type}"
            thor_action = {"action": "Pass"}
        
        if thor_action is None:
            thor_action = {"action": "Pass"}

        if self.agent_count > 1:
            aid = 0 if thor_agent_id is None else int(thor_agent_id)
            thor_action["agentId"] = aid
            print(f"  🤖 Embodied agent (agentId): {aid}")

        #       （       ）
        action_name_str = thor_action.get("action", "Unknown")
        params_list = []
        if "moveMagnitude" in thor_action:
            params_list.append(f"{thor_action['moveMagnitude']}")
        if "degrees" in thor_action:
            params_list.append(f"{thor_action['degrees']}")
        if "objectId" in thor_action:
            obj_id = thor_action["objectId"]
            actual_obj_type = obj_id.split("|")[0] if "|" in obj_id else "Object"
            
            #       ，       
            if action_name_str in ["PickupObject", "PutObject", "ThrowObject", "SliceObject", "BreakObject", 
                                    "CookObject", "OpenObject", "CloseObject", "DirtyObject", "ToggleObjectOn", 
                                    "ToggleObjectOff", "FillObjectWithLiquid", "EmptyLiquidFromObject"]:
                semantic_parent = None
                for parent, variants in SEMANTIC_OBJECT_MAPPING.items():
                    if actual_obj_type in variants:
                        semantic_parent = parent
                        break
                obj_type_to_record = semantic_parent if semantic_parent else actual_obj_type
            else:
                obj_type_to_record = actual_obj_type
            
            params_list.append(f"{obj_type_to_record}")
            
            if action_name_str == "FillObjectWithLiquid" and "fillLiquid" in thor_action:
                params_list.append(thor_action["fillLiquid"])
        
        params_str = ", ".join(params_list) if params_list else ""
        if params_str:
            self.action_sequence.append(f"{action_name_str}({params_str})")
        else:
            self.action_sequence.append(f"{action_name_str}()")
        
        #       （    ）
        action_params = thor_action.copy()
        
        #     
        try:
            event = self.controller.step(**action_params)
            error_message = None
            if not event.metadata.get("lastActionSuccess", True):
                error_message = event.metadata.get("errorMessage", "Unknown error")
                
                #      PutObject   
                if action_name == "PutObject" and error_message:
                    #            
                    if "cannot be placed" in error_message.lower() or "cannot place" in error_message.lower():
                        #              
                        inventory = event.metadata.get("inventoryObjects", [])
                        if inventory:
                            held_obj = inventory[0]
                            held_type = held_obj.get("objectType", "Unknown")
                            #       （   ID   ）
                            held_type_clean = held_type.split("|")[0] if "|" in held_type else held_type
                            
                            #       
                            container_type = object_type.split("|")[0] if object_type and "|" in object_type else object_type
                            
                            #      forceAction     
                            print(
                                "\n⚠️  PutObject failed; retrying with forceAction=True..."
                            )
                            action_params_forced = action_params.copy()
                            action_params_forced["forceAction"] = True
                            
                            try:
                                event_forced = self.controller.step(**action_params_forced)
                                if event_forced.metadata.get("lastActionSuccess", True):
                                    print("✓ PutObject succeeded with forceAction=True")
                                    event = event_forced
                                    error_message = None
                                else:
                                    # forceAction     
                                    error_message_forced = event_forced.metadata.get("errorMessage", "Unknown error")
                                    error_message = (
                                        f"⚠️  PutObject failed: {held_type_clean} could not be placed "
                                        f"into {container_type}, even with forceAction=True.\n"
                                        f"   Reason: {error_message_forced}\n"
                                        "   Hint: the object may be too far away, occluded, or not a valid receptacle."
                                    )
                            except Exception as e:
                                # forceAction     
                                error_message = (
                                    f"⚠️  PutObject failed: {held_type_clean} could not be placed "
                                    f"into {container_type}.\n"
                                    f"   Reason: forceAction retry raised {e}\n"
                                    "   Hint: try moving closer, rotating, or choosing another receptacle."
                                )
                        else:
                            #       ，     forceAction
                            error_message = (
                                f"⚠️  PutObject failed: no held object to place into {container_type}.\n"
                                "   Reason: inventoryObjects is empty.\n"
                                "   Hint: pick up an object before PutObject."
                            )
        except Exception as e:
            error_message = str(e)
            print(f"❌ Action execution error: {error_message}")
            event = self.controller.last_event

        #      （  ：         Event，    agent{k}/    ）
        if event is None:
            event = self.controller.last_event
        frame_agent_id = (
            thor_action.get("agentId") if self.agent_count > 1 else None
        )
        ev_unwrapped = self._unwrap_agent_event(event, thor_agent_id=frame_agent_id)
        aid_save = (
            (0 if thor_agent_id is None else int(thor_agent_id))
            if self.agent_count > 1
            else None
        )
        image_path = self._save_frame(
            ev_unwrapped.frame,
            prefix=f"step_{self.step_counter}",
            thor_agent_id=aid_save,
        )

        #       
        text_state = self._generate_text_state(event.metadata)

        #     
        if error_message is None:
            error_message = self._translate_error_message(
                event.metadata.get("errorMessage", ""), action_name, object_type
            ) if not event.metadata.get("lastActionSuccess", True) else None
        
        reward = self._compute_reward_from_metadata(
            event.metadata, action_name, error_message
        )
        
        #       
        done = self._check_done(event.metadata)
        
        observation = EnvObservation(
            image_path=image_path,
            text_state=text_state,
            reward=reward,
            done=done,
            metadata=event.metadata,
        )
        
        if error_message and event.metadata.get("lastActionSuccess", True):
            #             （     ），      
            error_message = None
        
        if error_message:
            print(f"⚠️  Action failed: {error_message}")
        
        return observation, error_message

    def _resolve_move_magnitude(
        self, magnitude: Optional[float] = None, granularity: Optional[str] = None
    ) -> float:
        """      ：         （Small/Medium/Large） 
        Small=0.25m, Medium=0.5m, Large=1m       Small 
          spatial-planning/envs/ai2thor/wrapper.py    
        """
        if magnitude is not None and isinstance(magnitude, (int, float)):
            return float(magnitude)
        if granularity:
            g = str(granularity).strip().lower()
            if g == "small":
                return self.move_small_magnitude
            if g == "medium":
                return self.move_medium_magnitude
            if g == "large":
                return self.move_large_magnitude
        return self.move_small_magnitude

    def _convert_navigation_action(
        self, action_name: str, magnitude: Optional[float] = None, degrees: Optional[float] = None
    ) -> dict:
        """      （           ）
          spatial-planning/envs/ai2thor/wrapper.py    

        Args:
            action_name:       
            magnitude:     （ ），   MoveAhead/Back/Left/Right
            degrees:   /    ，   RotateLeft/Right, LookUp/Down
        """
        move_mag = magnitude if magnitude is not None else self.move_small_magnitude
        rot_deg = float(degrees) if degrees is not None else self.rotate_degrees
        look_deg = float(degrees) if degrees is not None else 30
        action_map = {
            "MoveAhead": {"action": "MoveAhead", "moveMagnitude": move_mag},
            "MoveBack": {"action": "MoveBack", "moveMagnitude": move_mag},
            "MoveLeft": {"action": "MoveLeft", "moveMagnitude": move_mag},
            "MoveRight": {"action": "MoveRight", "moveMagnitude": move_mag},
            "RotateLeft": {"action": "RotateLeft", "degrees": rot_deg},
            "RotateRight": {"action": "RotateRight", "degrees": rot_deg},
            "LookUp": {"action": "LookUp", "degrees": look_deg},
            "LookDown": {"action": "LookDown", "degrees": look_deg},
            "Crouch": {"action": "Crouch"},
            "Stand": {"action": "Stand"},
        }
        return action_map.get(action_name, {"action": "Pass"})
    
    def _convert_interaction_action(
        self, action_name: str, object_type: Optional[str]
    ) -> Tuple[Optional[dict], Optional[str]]:
        """      （      ）
        
           spatial-planning/envs/ai2thor/wrapper.py
        
        For PickupObject:      SEMANTIC_OBJECT_MAPPING      
        """
        event = self.controller.last_event
        
        # DropHandObject      
        if action_name == "DropHandObject":
            inventory = event.metadata.get("inventoryObjects", [])
            if not inventory:
                return None, "Hand is empty, cannot drop"
            return {"action": "DropHandObject", "forceAction": False}, None
        
        #               
        if not object_type:
            return None, f"{action_name} requires object type specification"
        
        #     ：   PickupObject，      
        if action_name == "PickupObject":
            candidate_types = SEMANTIC_OBJECT_MAPPING.get(object_type, [object_type])
            all_objects = event.metadata["objects"]
            
            best_obj = None
            min_dist = float("inf")
            
            for obj in all_objects:
                if obj["objectType"] in candidate_types and obj["visible"]:
                    if obj["distance"] < min_dist:
                        min_dist = obj["distance"]
                        best_obj = obj
            
            if best_obj:
                object_id = best_obj["objectId"]
                actual_type = best_obj["objectType"]
                
                if actual_type != object_type:
                    print(f"📦 Semantic search: PickupObject({object_type}) -> actually picking {actual_type}")
                
                return {
                    "action": "PickupObject",
                    "objectId": object_id,
                    "forceAction": False,
                }, None
            else:
                invisible_objects = [obj for obj in all_objects if obj["objectType"] in candidate_types]
                if invisible_objects:
                    nearest_obj = min(invisible_objects, key=lambda obj: obj["distance"])
                    distance = nearest_obj.get("distance", 0)
                    return None, f"{object_type} is not in view, need to approach or adjust view (distance: {distance:.1f}m)"
                else:
                    return None, f"{object_type} (or its transformed variants) does not exist in scene"
        
        #       ：        
        all_objects = [
            obj for obj in event.metadata["objects"]
            if obj["objectType"] == object_type or obj["objectType"].startswith(object_type + "|")
        ]
        
        if not all_objects:
            return None, f"{object_type} does not exist in scene"
        
        visible_objects = [obj for obj in all_objects if obj["visible"]]
        
        if not visible_objects:
            nearest_obj = min(all_objects, key=lambda obj: obj["distance"])
            distance = nearest_obj.get("distance", 0)
            return None, f"{object_type} is not in view, need to approach or adjust view (distance: {distance:.1f}m)"
        
        target_obj = min(visible_objects, key=lambda obj: obj["distance"])
        object_id = target_obj["objectId"]
        
        #         AI2-THOR   
        action_map = {
            "PickupObject": "PickupObject",
            "PutObject": "PutObject",
            "ThrowObject": "ThrowObject",
            "OpenObject": "OpenObject",
            "CloseObject": "CloseObject",
            "ToggleObjectOn": "ToggleObjectOn",
            "ToggleObjectOff": "ToggleObjectOff",
            "SliceObject": "SliceObject",
            "BreakObject": "BreakObject",
            "CookObject": "CookObject",
            "DirtyObject": "DirtyObject",
            "CleanObject": "CleanObject",
            "FillObjectWithLiquid": "FillObjectWithLiquid",
            "EmptyLiquidFromObject": "EmptyLiquidFromObject",
            "UseUpObject": "UseUpObject",
        }
        
        thor_action_name = action_map.get(action_name)
        if not thor_action_name:
            return None, f"Unknown interaction action: {action_name}"
        
        thor_action = {
            "action": thor_action_name,
            "objectId": object_id,
            "forceAction": False,
        }
        
        # FillObjectWithLiquid         （   water）
        if action_name == "FillObjectWithLiquid":
            thor_action["fillLiquid"] = "water"
        
        return thor_action, None
    
    def _translate_error_message(
        self, raw_error: str, action_name: str, object_type: Optional[str]
    ) -> str:
        """   AI2-THOR           
        
           spatial-planning/envs/ai2thor/wrapper.py
        """
        error_lower = raw_error.lower()
        obj = object_type or "object"
        
        # DropHandObject     （           ）
        if "can't be dropped" in error_lower and ("collision" in error_lower or "agent" in error_lower):
            return "⚠️ DropHandObject failed because the drop path is blocked. Try moving away with MoveAhead/MoveBack/MoveLeft/MoveRight, then DropHandObject again; or use ThrowObject with low force."
        
        #       
        if "hand has something" in error_lower or "already holding" in error_lower:
            return "Agent is already holding an object; drop it first with DropHandObject"
        elif "hand is empty" in error_lower:
            return "Agent hand is empty; pick up an object first"
        elif "can't be dropped" in error_lower and "collision" in error_lower:
            return "Drop failed because the target area is blocked. Move to a clearer spot or use ThrowObject with low force."
        
        #         
        elif "not in range" in error_lower or "too far" in error_lower:
            return f"Too far from {obj}; move closer first"
        elif "not visible" in error_lower:
            return f"{obj} is not visible; look around or move closer"
        elif "out of reach" in error_lower:
            return f"Cannot reach {obj}; move closer first"
        
        #       
        elif "not interactable" in error_lower:
            return f"{obj} is not interactable"
        elif "not pickupable" in error_lower or "can't pickup" in error_lower:
            return f"{obj} is not pickupable"
        elif "not receptacle" in error_lower:
            return f"{obj} is not a receptacle; choose another target"
        
        #     
        elif "not openable" in error_lower or "can't open" in error_lower:
            return f"{obj} is not openable"
        elif "already open" in error_lower:
            return f"{obj} is already in the requested state"
        elif "already closed" in error_lower:
            return f"{obj} is already in the requested state"
        elif "not toggleable" in error_lower:
            return f"{obj} is not toggleable"
        elif "already on" in error_lower or "already powered on" in error_lower:
            return f"{obj} is already in the requested state"
        elif "already off" in error_lower or "already powered off" in error_lower:
            return f"{obj} is already in the requested state"
        
        #       
        elif "not sliceable" in error_lower:
            return f"{obj} is not sliceable"
        elif "already sliced" in error_lower:
            return f"{obj} is already in the requested state"
        elif "not breakable" in error_lower:
            return f"{obj} is already in the requested state"
        elif "not cookable" in error_lower:
            return f"{obj} is already in the requested state"
        elif "not cleanable" in error_lower:
            return f"{obj} is already in the requested state"
        elif "not fillable" in error_lower:
            return f"{obj} cannot be filled with liquid"
        
        #      
        elif "no object" in error_lower or "object not found" in error_lower:
            return f"Could not find {obj} in the scene"
        elif "no such object" in error_lower:
            return f"No such object: {obj}"
        
        #        
        elif "path blocked" in error_lower or "collision" in error_lower:
            return "Path is blocked; try moving or rotating first"
        elif "can't move" in error_lower or "cannot move" in error_lower:
            return "Cannot move in that direction"
        
        #       
        elif "invalid action" in error_lower:
            return f"Invalid action: {action_name}"
        elif "failed" in error_lower:
            return f"Action {action_name} failed: {raw_error}"
        
        #     ，      
        else:
            return f"[Environment error] {raw_error}"
    
    def _compute_reward_from_metadata(
        self, metadata: dict, action_name: str, error_message: Optional[str]
    ) -> float:
        """              
        
           spatial-planning/envs/ai2thor/wrapper.py
        """
        #       
        if error_message:
            return self.step_failure_penalty
        
        #         
        if metadata["lastActionSuccess"]:
            return self.step_success_bonus
        
        return self.step_failure_penalty
    
    def _check_done(self, metadata: dict) -> bool:
        """        （    ，      ）
        
           spatial-planning/envs/ai2thor/wrapper.py
        """
        #          ，   False
        if not self.target_object_types or not self.success_predicate:
            return False
        
        #     ：object_in_receptacle   
        if self.success_condition and self.success_condition.get("type") == "object_in_receptacle":
            object_type = self.success_condition.get("object_type", "Apple")
            receptacle_type = self.success_condition.get("receptacle_type", "Plate")
            
            for obj in metadata["objects"]:
                obj_type = obj["objectType"]
                if obj_type == object_type or obj_type.startswith(object_type + "|"):
                    if self.success_predicate(obj):
                        print(f"  ✓ Task completion condition met: {object_type} placed in {receptacle_type}")
                        return True
            return False
        
        #     ：object_in_hand   
        if self.success_condition and self.success_condition.get("type") == "object_in_hand":
            target_type = self.success_condition.get("object_type")
            inventory = metadata.get("inventoryObjects", [])
            for item in inventory:
                item_type = item["objectType"]
                if item_type == target_type or item_type.startswith(target_type + "|"):
                    print(f"  ✓ Task completion condition met: {target_type} in hand")
                    return True
            return False
        
        #     ：         
        for obj in metadata["objects"]:
            if obj["objectType"] in self.target_object_types:
                if self.success_predicate(obj):
                    print(f"  ✓ Task completion condition met: {obj['objectType']} meets success criteria")
                    return True
        
        return False
    
    def _get_visible_objects(self):
        """           
        
        Returns:
            List[Dict]:       ，     
        """
        if self.controller is None:
            return []
        
        event = self.controller.last_event
        metadata = event.metadata
        
        #       
        visible_objects = [obj for obj in metadata.get("objects", []) if obj.get("visible", False)]
        
        #      
        visible_objects.sort(key=lambda x: x.get("distance", float("inf")))
        
        return visible_objects
    
    def _get_current_observation(self) -> EnvObservation:
        """       （   reset）

            ：    agent0 agent1    ``reset_*.png``   ``agent1/`` ``agent2/``    ，
           agent0          observation    
            ：    ``controller.last_event``          
        """
        if self.controller is None:
            raise RuntimeError("Environment is not initialized; call reset() first")

        if self.agent_count > 1:
            #     agent0    （prefix="reset"），     agent1        
            ev0_raw = self.controller.step(action="Pass", agentId=0)
            ev0 = self._unwrap_agent_event(ev0_raw, thor_agent_id=0)
            image_path = self._save_frame(ev0.frame, prefix="reset", thor_agent_id=0)
            try:
                ev1_raw = self.controller.step(action="Pass", agentId=1)
                ev1 = self._unwrap_agent_event(ev1_raw, thor_agent_id=1)
                self._save_frame(ev1.frame, prefix="reset", thor_agent_id=1)
            except Exception as e:
                print(f"⚠️  Failed to save agent2 reset frame: {e}")
            metadata = ev0_raw.metadata
        else:
            event = self.controller.last_event
            image_path = self._save_frame(event.frame, prefix="reset")
            metadata = event.metadata

        text_state = self._generate_text_state(metadata)
        observation = EnvObservation(
            image_path=image_path,
            text_state=text_state,
            reward=0.0,
            done=False,
            metadata=metadata,
        )
        return observation
    
    def _generate_text_state(self, metadata: Dict[str, Any]) -> str:
        """        
        
        Args:
            metadata: AI2-THOR    
            
        Returns:
            str:       
        """
        #             （    first_person）
        env_config = self.config.get("env", {})
        text_state_mode = env_config.get("text_state_mode", "first_person")
        
        #        
        agent = metadata.get("agent", {})
        position = agent.get("position", {})
        rotation = agent.get("rotation", {})
        
        #     
        parts = []
        parts.append("ProcTHOR Environment")
        
        if text_state_mode == "omniscient":
            #     ：      
            pos_str = f"({position.get('x', 0.0):.2f}, {position.get('y', 0.9):.2f}, {position.get('z', 0.0):.2f})"
            parts.append(f"Position: {pos_str}")
            parts.append(f"Rotation: {rotation.get('y', 0.0):.1f}°")
            
            #       
            visible_objs = self._get_visible_objects()
            if visible_objs:
                obj_types = [obj.get("objectType", "Unknown") for obj in visible_objs[:10]]  #    10 
                obj_types_str = ", ".join(obj_types)
                if len(visible_objs) > 10:
                    obj_types_str += f" ... (  {len(visible_objs)}  )"
                parts.append(f"Visible objects: [{obj_types_str}]")
        else:
            #       ：       
            if self.task_description:
                parts.append(f"Task: {self.task_description}")
            
            #             
            last_action_success = metadata.get("lastActionSuccess", True)
            if not last_action_success:
                error_msg = metadata.get("errorMessage", "      ")
                parts.append(f"Last action failed: {error_msg}")
        
        if self.text_state_mode == "omniscient":
            return self._generate_omniscient_text_state(metadata)
        return self._generate_first_person_text_state(metadata)
    
    def _generate_first_person_text_state(self, metadata: Dict[str, Any]) -> str:
        """          
        
           spatial-planning/envs/ai2thor/wrapper.py
        """
        lines = [
            " First-Person Mode Please mainly rely on images to judge the environment, no additional global state provided.",
            f"Task description: {self.task_description}",
            f"Current step: {self.step_counter}",
            f"Last action success: {'Yes' if metadata['lastActionSuccess'] else 'No'}",
        ]
        if not metadata["lastActionSuccess"]:
            error_msg = metadata.get("errorMessage", "Unknown")
            lines.append(f"Error message: {error_msg}")
            if "collision" in error_msg.lower() or "block" in error_msg.lower() or "obstacle" in error_msg.lower():
                lines.append("Hint: There may be an obstacle ahead, try interacting or changing direction to bypass.")
            elif "out of reach" in error_msg.lower() or "too far" in error_msg.lower():
                lines.append("Hint: Target may be too far, please approach first before interacting.")
            elif "not visible" in error_msg.lower():
                lines.append("Hint: Target not found in current view, please rotate or move to search again.")
        return "\n".join(lines)
    
    def _generate_omniscient_text_state(self, metadata: Dict[str, Any]) -> str:
        """        
        
           spatial-planning/envs/ai2thor/wrapper.py
        """
        agent = metadata["agent"]
        position = agent["position"]
        rotation = agent["rotation"]
        
        visible_objects = [
            obj["objectType"] for obj in metadata["objects"] if obj["visible"]
        ]
        visible_objects = sorted(set(visible_objects))
        
        if len(visible_objects) > 10:
            visible_objects_str = ", ".join(visible_objects[:10]) + f", ... (total {len(visible_objects)} objects)"
        else:
            visible_objects_str = ", ".join(visible_objects) if visible_objects else "None"
        
        text_state = f"""Scene: ProcTHOR (scene_index={self.scene_index})
Agent position: ({position["x"]:.2f}, {position["y"]:.2f}, {position["z"]:.2f})
Agent heading: {rotation["y"]:.1f}°
Visible objects: {visible_objects_str}"""
        
        if self.target_object_types:
            task_summary = self._generate_task_status_summary(metadata)
            text_state += f"\n\n Task Status \n{task_summary}"
        
        text_state += f"\n\nLast action success: {'Yes' if metadata['lastActionSuccess'] else 'No'}"
        
        if not metadata["lastActionSuccess"]:
            error_msg = metadata.get("errorMessage", "Unknown")
            text_state += f"\nError message: {error_msg}"
            
            if "collision" in error_msg.lower() or "block" in error_msg.lower() or "obstacle" in error_msg.lower():
                text_state += "\n⚠️  Movement blocked! Obstacle ahead. Suggestions:"
                text_state += "\n   1. If the object ahead is the target, try interacting with it (interact=true)"
                text_state += "\n   2. Otherwise rotate view to change direction, or back up/side step to bypass"
                text_state += "\n   3. Don't continuously move in the same direction"
            elif "out of reach" in error_msg.lower() or "too far" in error_msg.lower():
                text_state += "\n⚠️  Object too far! Please approach target object first before interacting"
            elif "not visible" in error_msg.lower():
                text_state += "\n⚠️  Target not visible! Please rotate view or move to find target"
        
        return text_state
    
    def step(self, action: EnvAction) -> EnvObservation:
        """      （   EnvAction   ）
        
           spatial-planning/envs/ai2thor/wrapper.py
        
        Args:
            action:     
            
        Returns:
                 
        """
        self.step_counter += 1
        
        #        
        thor_actions = self._convert_action(action)
        
        #     
        print(f"\n--- Step {self.step_counter} ---")
        if action.comment:
            print(f"💭 Thinking: {action.comment}")
        print(f"🎬 Action: {self._format_action(action)}")
        
        #       
        for thor_action in thor_actions:
            action_name = thor_action.get("action", "Unknown")
            params_list = []
            if "moveMagnitude" in thor_action:
                params_list.append(f"{thor_action['moveMagnitude']}")
            if "degrees" in thor_action:
                params_list.append(f"{thor_action['degrees']}")
            if "objectId" in thor_action:
                obj_id = thor_action["objectId"]
                obj_type = obj_id.split("|")[0] if "|" in obj_id else "Object"
                params_list.append(f"{obj_type}")
            
            params_str = ", ".join(params_list) if params_list else ""
            if params_str:
                self.action_sequence.append(f"{action_name}({params_str})")
            else:
                self.action_sequence.append(f"{action_name}()")
        
        #           
        event = None
        for thor_action in thor_actions:
            event = self.controller.step(**thor_action)
            if not event.metadata["lastActionSuccess"]:
                print(f"⚠️  Action failed: {event.metadata.get('errorMessage', 'Unknown error')}")
                break
        
        #      
        if event is None:
            event = self.controller.last_event
        image_path = self._save_frame(event.frame, prefix=f"step_{self.step_counter}")
        
        #       
        text_state = self._generate_text_state(event.metadata)
        
        #     
        reward = self._compute_reward(action, event.metadata)
        
        #       
        done = self._check_done(event.metadata)
        
        return EnvObservation(
            image_path=image_path,
            text_state=text_state,
            reward=reward,
            done=done,
            metadata=event.metadata,
        )
    
    def _convert_action(self, action: EnvAction) -> List[dict]:
        """   EnvAction   AI2-THOR     
        
           spatial-planning/envs/ai2thor/wrapper.py
        """
        #          ，    
        if action.thor_actions and len(action.thor_actions) > 0:
            ta = action.thor_actions[0]
            print(f"  ✓ Using low-level action: {ta.name}")
            
            #        objectId      
            object_actions = [
                "OpenObject", "CloseObject", "ToggleObjectOn", "ToggleObjectOff",
                "PickupObject", "PutObject", "SliceObject", "BreakObject",
                "CookObject", "DirtyObject", "CleanObject", "FillObjectWithLiquid",
                "EmptyLiquidFromObject", "UseUpObject",
            ]
            
            if ta.name in object_actions:
                target_obj = self._find_interaction_target(ta.name)
                if target_obj:
                    thor_action = {"action": ta.name, "objectId": target_obj["objectId"]}
                    if ta.name == "FillObjectWithLiquid":
                        fill_liquid = ta.args.get("fillLiquid", "water")
                        thor_action["fillLiquid"] = fill_liquid
                    return [thor_action]
                else:
                    return [{"action": "Pass"}]
            elif ta.name in ["DropHandObject", "ThrowObject"]:
                thor_action = {"action": ta.name}
                if ta.name == "ThrowObject":
                    thor_action["moveMagnitude"] = ta.args.get("moveMagnitude", 150.0)
                return [thor_action]
            else:
                thor_action = {"action": ta.name, **ta.args}
                return [thor_action]
        
        #     ：            
        print(f"  ℹ️  Using traditional action conversion logic (move/turn/interact)")
        return self._convert_from_move_turn_interact(action)
    
    def _convert_from_move_turn_interact(self, action: EnvAction) -> List[dict]:
        """  move/turn/interact       
        
           spatial-planning/envs/ai2thor/wrapper.py
        """
        thor_actions = []
        
        #     
        if action.turn is not None:
            if action.turn > 0:
                thor_actions.append({"action": "RotateRight", "degrees": abs(action.turn)})
            else:
                thor_actions.append({"action": "RotateLeft", "degrees": abs(action.turn)})
        
        #     
        if action.move:
            move_map = {
                "forward": {"action": "MoveAhead", "moveMagnitude": self.move_ahead_magnitude},
                "back": {"action": "MoveBack", "moveMagnitude": self.move_back_magnitude},
                "left": {"action": "MoveLeft", "moveMagnitude": self.move_left_magnitude},
                "right": {"action": "MoveRight", "moveMagnitude": self.move_right_magnitude},
            }
            if action.move in move_map:
                thor_actions.append(move_map[action.move])
        
        #     
        if action.interact:
            interaction_action = self._determine_interaction_action()
            if interaction_action:
                thor_actions.append(interaction_action)
        
        return thor_actions if thor_actions else [{"action": "Pass"}]
    
    def _determine_interaction_action(self) -> Optional[dict]:
        """          
        
           spatial-planning/envs/ai2thor/wrapper.py
        """
        event = self.controller.last_event
        visible_objects = [obj for obj in event.metadata["objects"] if obj["visible"]]
        
        if not visible_objects:
            return None
        
        #             
        target_obj = min(visible_objects, key=lambda obj: obj["distance"])
        
        #           
        if target_obj.get("openable"):
            return {
                "action": "OpenObject" if not target_obj.get("isOpen") else "CloseObject",
                "objectId": target_obj["objectId"],
            }
        elif target_obj.get("toggleable"):
            return {
                "action": "ToggleObjectOn" if not target_obj.get("isToggled") else "ToggleObjectOff",
                "objectId": target_obj["objectId"],
            }
        elif target_obj.get("pickupable") and not target_obj.get("isPickedUp"):
            return {"action": "PickupObject", "objectId": target_obj["objectId"]}
        
        return None
    
    def _find_interaction_target(self, action_name: str) -> Optional[dict]:
        """             
        
           spatial-planning/envs/ai2thor/wrapper.py
        """
        event = self.controller.last_event
        
        #   1：          
        target_candidates = []
        if self.target_object_types:
            target_candidates = [
                obj for obj in event.metadata["objects"]
                if obj["visible"] and obj["objectType"] in self.target_object_types
            ]
        
        #   2：        ，         
        if not target_candidates:
            if action_name in ["OpenObject", "CloseObject"]:
                target_candidates = [
                    obj for obj in event.metadata["objects"]
                    if obj["visible"] and obj.get("openable", False)
                ]
            elif action_name in ["ToggleObjectOn", "ToggleObjectOff"]:
                target_candidates = [
                    obj for obj in event.metadata["objects"]
                    if obj["visible"] and obj.get("toggleable", False)
                ]
            elif action_name == "PickupObject":
                target_candidates = [
                    obj for obj in event.metadata["objects"]
                    if obj["visible"] and obj.get("pickupable", False)
                ]
            elif action_name == "PutObject":
                target_candidates = [
                    obj for obj in event.metadata["objects"]
                    if obj["visible"] and obj.get("receptacle", False)
                ]
            #       ...
        
        if not target_candidates:
            return None
        
        #             
        if action_name == "OpenObject":
            closed = [obj for obj in target_candidates if not obj.get("isOpen", False)]
            target_candidates = closed if closed else target_candidates
        elif action_name == "CloseObject":
            open_objs = [obj for obj in target_candidates if obj.get("isOpen", False)]
            target_candidates = open_objs if open_objs else target_candidates
        
        #        
        return min(target_candidates, key=lambda obj: obj["distance"])
    
    def _format_action(self, action: EnvAction) -> str:
        """         
        
           spatial-planning/envs/ai2thor/wrapper.py
        """
        parts = []
        if action.move:
            parts.append(f"move={action.move}")
        if action.turn is not None:
            direction = "right" if action.turn > 0 else "left"
            parts.append(f"{direction} {abs(action.turn)}°")
        if action.interact:
            parts.append("interact=True")
        
        return ", ".join(parts) if parts else "No action"
    
    def _compute_reward(self, action: EnvAction, metadata: dict) -> float:
        """    （    ）
        
           spatial-planning/envs/ai2thor/wrapper.py
        """
        if not metadata["lastActionSuccess"]:
            return self.step_failure_penalty
        
        #         
        if self._check_done(metadata):
            return self.success_reward
        
        #         
        return self.step_success_bonus
    
    @staticmethod
    def _thor_agent_image_subdir(thor_agent_id: int) -> str:
        """Multi-agent: map THOR agentId (0,1,...) to image subdir (agent1, agent2, ...)."""
        return f"agent{int(thor_agent_id) + 1}"

    def _save_frame(
        self,
        frame: Optional[np.ndarray],
        prefix: str = "frame",
        thor_agent_id: Optional[int] = None,
    ) -> str:
        """     ；   spatial-planning/envs/ai2thor/wrapper.py 

          ：``<output_dir>/{prefix}_{timestamp}.png`` 
          ：``<output_dir>/agent{k}/{prefix}_{timestamp}.png``（k=1   agentId=0） 
        """
        if frame is None:
            raise RuntimeError(
                "event.frame is None, so RGB cannot be saved: AI2-THOR Controller may be running with headless=True, "
                "renderImage=False, or ProcTHOR may lack DISPLAY/CloudRendering support. "
                "Use headless=False or set runtime_headless before reset()."
            )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.png"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.agent_count > 1 and thor_agent_id is not None:
            sub = self._thor_agent_image_subdir(thor_agent_id)
            out_dir = self.output_dir / sub
            out_dir.mkdir(parents=True, exist_ok=True)
            filepath = out_dir / filename
        else:
            filepath = self.output_dir / filename

        image = Image.fromarray(frame)
        image.save(filepath)

        return str(filepath)
    
    def _generate_task_status_summary(self, metadata: Dict[str, Any]) -> str:
        """        
        
           spatial-planning/envs/ai2thor/wrapper.py
        """
        if not self.target_object_types or not self.success_predicate:
            return "No task configured"
        
        target_objects = [
            obj for obj in metadata["objects"] if obj["objectType"] in self.target_object_types
        ]
        
        if not target_objects:
            return f"Target objects ({', '.join(self.target_object_types)}) not found in scene"
        
        status_lines = [f"Target: {self.target_description}"]
        for obj in target_objects[:5]:  #      5 
            obj_type = obj["objectType"]
            satisfies = self.success_predicate(obj)
            status_lines.append(f"  - {obj_type}: {'✓' if satisfies else '✗'}")
        
        if len(target_objects) > 5:
            status_lines.append(f"  ... (and {len(target_objects) - 5} more)")
        
        return "\n".join(status_lines)
    
    def close(self):
        """    """
        if self.controller is not None:
            self.controller.stop()
            self.controller = None
        print("✓ ProcTHOR closed")
