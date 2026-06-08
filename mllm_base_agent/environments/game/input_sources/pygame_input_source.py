"""
Pygame       
"""
import os
import pygame
import numpy as np
from typing import Optional, Dict, Any
import time

from core.input_source import GameInputSource
from core.data_classes import FrameData, GameState, Action


class PygameInputSource(GameInputSource):
    """Pygame     """

    def __init__(self):
        super().__init__()
        self.screen = None
        self.game_module = None
        self.frame_count = 0
        self.is_headless = False
        self.screen_size = (800, 600)  #       

    def _get_capabilities(self) -> Dict[str, bool]:
        """  Pygame        """
        return {
            "real_time": True,
            "supports_actions": True,
            "supports_state_reading": True,
            "supports_reset": True,
            "headless": True,
        }

    def initialize(self, game_module=None, headless: bool = False,
                   screen_size: tuple = (800, 600), **kwargs) -> bool:
        """
           Pygame  

        Args:
            game_module:       ，    init/update/render/get_state/reset  
            headless:         
            screen_size:      (width, height)
            **kwargs:     

        Returns:
            bool:        
        """
        try:
            self.is_headless = headless
            self.screen_size = screen_size

            #           
            if headless:
                os.environ["SDL_VIDEODRIVER"] = "dummy"
                os.environ["SDL_AUDIODRIVER"] = "dummy"

            #    pygame
            pygame.init()

            if headless:
                self.screen = pygame.Surface(screen_size)
            else:
                self.screen = pygame.display.set_mode(screen_size)
                pygame.display.set_caption("MLLM Game Evaluation")

            #       
            if game_module:
                self.game_module = game_module
                if hasattr(game_module, 'init'):
                    game_module.init()
                return True

            return True

        except Exception as e:
            print(f"Failed to initialize Pygame: {e}")
            return False

    def capture_frame(self) -> Optional[FrameData]:
        """       """
        try:
            #       ，            
            if self.is_headless and self.game_module:
                #     
                self.screen.fill((0, 0, 0))  #     

                #           ，          
                if hasattr(self.game_module, 'render'):
                    #   render      screen  
                    import inspect
                    sig = inspect.signature(self.game_module.render)
                    if 'screen' in sig.parameters:
                        self.game_module.render(screen=self.screen)
                    else:
                        self.game_module.render()

                #                     
                if not hasattr(self.game_module, 'render'):
                    self._render_default_game()
            else:
                #       ，      
                if self.game_module and hasattr(self.game_module, 'render'):
                    self.game_module.render()

            #         
            if self.is_headless:
                surface = self.screen
            else:
                surface = pygame.display.get_surface()

            if surface is None:
                return None

            #  pygame     numpy  
            #   pygame.surfarray.array3d  RGB     ，      
            image_array = pygame.surfarray.array3d(surface)

            #       (W, H, C)   (H, W, C)
            image_array = np.transpose(image_array, (1, 0, 2))

            #        
            frame_data = FrameData(
                image=image_array,
                timestamp=time.time(),
                frame_number=self.frame_count,
                metadata={
                    "screen_size": self.screen_size,
                    "headless": self.is_headless
                }
            )

            self.frame_count += 1
            return frame_data

        except Exception as e:
            print(f"Failed to capture frame: {e}")
            return None

    def execute_action(self, action: Action) -> bool:
        """    """
        try:
            #       ，                 
            if self.is_headless and self.game_module:
                #          
                action_result = True
                repeat = self._get_action_repeat(action)
                any_success = False
                if action.type == "key_press":
                    for _ in range(repeat):
                        #              
                        if hasattr(self.game_module, 'execute_mapped_action'):
                            action_result = self.game_module.execute_mapped_action(action.key)
                        elif hasattr(self.game_module, 'get_action_mapping'):
                            #        ，       
                            try:
                                from core.action_mapping import ActionMapping
                                mapping = self.game_module.get_action_mapping(action.key)
                                if mapping and mapping.method_name:
                                    method = getattr(self.game_module, mapping.method_name, None)
                                    if method and callable(method):
                                        action_result = method()
                                    else:
                                        action_result = False
                                else:
                                    action_result = False
                            except ImportError:
                                #          
                                action_result = self._execute_fallback_action(action.key)
                        else:
                            #          
                            action_result = self._execute_fallback_action(action.key)

                        any_success = any_success or bool(action_result)

                        #        repeat>1               
                        if hasattr(self.game_module, 'update'):
                            self.game_module.update()

                        if not action_result or self._game_has_finished():
                            break

                    action_result = any_success
                else:
                    #       
                    if hasattr(self.game_module, 'update'):
                        self.game_module.update()

                return action_result
            else:
                #       ，      
                if action.type == "key_press":
                    key_event = self._key_to_pygame_event(action.key, "press")
                    if key_event:
                        pygame.event.post(key_event)
                elif action.type == "key_release":
                    key_event = self._key_to_pygame_event(action.key, "release")
                    if key_event:
                        pygame.event.post(key_event)

                #       
                if self.game_module and hasattr(self.game_module, 'update'):
                    self.game_module.update()

                return True

        except Exception as e:
            print(f"Failed to execute action: {e}")
            return False

    def _get_action_repeat(self, action: Action) -> int:
        """               ，       """
        repeat = 1
        if action.metadata:
            repeat = action.metadata.get("repeat", 1)

        try:
            repeat = int(repeat)
        except (TypeError, ValueError):
            repeat = 1

        return max(1, min(repeat, 10))

    def _game_has_finished(self) -> bool:
        """          ，           """
        if not self.game_module or not hasattr(self.game_module, 'get_state'):
            return False

        try:
            state = self.game_module.get_state()
        except Exception:
            return False

        if not isinstance(state, dict):
            return False

        return bool(state.get("game_over") or state.get("game_won") or state.get("success"))

    def get_game_state(self) -> Optional[GameState]:
        """      """
        try:
            if self.game_module and hasattr(self.game_module, 'get_state'):
                raw_state = self.game_module.get_state()

                #      （            ）
                normalized_state = self._normalize_game_state(raw_state)

                return GameState(
                    raw_state=raw_state,
                    normalized_state=normalized_state
                )
            return None
        except Exception as e:
            print(f"Failed to get game state: {e}")
            return None

    def reset_game(self) -> bool:
        """    """
        try:
            if self.game_module and hasattr(self.game_module, 'reset'):
                self.game_module.reset()
                self.frame_count = 0
                return True
            return False
        except Exception as e:
            print(f"Failed to reset game: {e}")
            return False

    def close(self):
        """         """
        try:
            pygame.quit()
        except Exception as e:
            print(f"Error during pygame quit: {e}")

    def _key_to_pygame_event(self, key: str, event_type: str) -> Optional[pygame.event.Event]:
        """      pygame  """
        key_mapping = {
            "w": pygame.K_w,
            "a": pygame.K_a,
            "s": pygame.K_s,
            "d": pygame.K_d,
            "q": pygame.K_q,
            "e": pygame.K_e,
            "space": pygame.K_SPACE,
            "up": pygame.K_UP,
            "down": pygame.K_DOWN,
            "left": pygame.K_LEFT,
            "right": pygame.K_RIGHT,
        }

        if key not in key_mapping:
            return None

        pygame_key = key_mapping[key]

        if event_type == "press":
            return pygame.event.Event(pygame.KEYDOWN, key=pygame_key)
        elif event_type == "release":
            return pygame.event.Event(pygame.KEYUP, key=pygame_key)

        return None

    def _mouse_to_pygame_event(self, pos: tuple, button: Optional[int],
                              event_type: str) -> Optional[pygame.event.Event]:
        """        pygame  """
        if event_type == "click":
            if button == 0:  #   
                return pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                        pos=pos, button=1)
            elif button == 1:  #   
                return pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                        pos=pos, button=2)
            elif button == 2:  #   
                return pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                        pos=pos, button=3)
        elif event_type == "move":
            return pygame.event.Event(pygame.MOUSEMOTION, pos=pos)

        return None

    def _render_default_game(self):
        """        """
        try:
            if not self.game_module:
                return

            #       
            state = self.game_module.get_state() if hasattr(self.game_module, 'get_state') else {}

            #     
            if 'player_pos' in state:
                pos = state['player_pos']
                pygame.draw.circle(self.screen, (255, 0, 0), (int(pos[0]), int(pos[1])), 20)

            #      -            
            if hasattr(self.game_module, 'targets'):
                for target in self.game_module.targets:
                    if 'pos' in target:
                        pos = target['pos']
                        pygame.draw.circle(self.screen, (0, 255, 0), (int(pos[0]), int(pos[1])), 15)

            #      -            
            if hasattr(self.game_module, 'enemies'):
                for enemy in self.game_module.enemies:
                    if 'pos' in enemy:
                        pos = enemy['pos']
                        pygame.draw.circle(self.screen, (0, 0, 255), (int(pos[0]), int(pos[1])), 15)

            #         
            font = pygame.font.Font(None, 36)
            if 'score' in state:
                score_text = font.render(f"Score: {state['score']}", True, (255, 255, 255))
                self.screen.blit(score_text, (10, 10))

            if 'health' in state:
                health_text = font.render(f"Health: {state['health']}", True, (255, 255, 255))
                self.screen.blit(health_text, (10, 50))

            if 'level' in state:
                level_text = font.render(f"Level: {state['level']}", True, (255, 255, 255))
                self.screen.blit(level_text, (10, 90))

        except Exception as e:
            print(f"Error in default game rendering: {e}")

    def _execute_fallback_action(self, key: str) -> bool:
        """      （     ）"""
        action_result = True
        if key == "w":  #    (Y  )
            if hasattr(self.game_module, 'move_up'):
                action_result = self.game_module.move_up()
            elif hasattr(self.game_module, 'move_forward'):
                action_result = self.game_module.move_forward()
        elif key == "s":  #    (Y  )
            if hasattr(self.game_module, 'move_down'):
                action_result = self.game_module.move_down()
            elif hasattr(self.game_module, 'move_backward'):
                action_result = self.game_module.move_backward()
        elif key == "a":  #    (X  )
            if hasattr(self.game_module, 'move_left'):
                action_result = self.game_module.move_left()
            elif hasattr(self.game_module, 'turn_left'):
                self.game_module.turn_left()
        elif key == "d":  #    (X  )
            if hasattr(self.game_module, 'move_right'):
                action_result = self.game_module.move_right()
            elif hasattr(self.game_module, 'turn_right'):
                self.game_module.turn_right()
        elif key == "q":  #    (Z  ) -      
            if hasattr(self.game_module, 'move_forward'):
                action_result = self.game_module.move_forward()
            else:
                action_result = False
        elif key == "e":  #    (Z  ) -      
            if hasattr(self.game_module, 'move_backward'):
                action_result = self.game_module.move_backward()
            else:
                action_result = False
        else:
            action_result = False

        return action_result

    def _normalize_game_state(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        """       """
        normalized = {}

        #         
        if isinstance(raw_state, dict):
            #           
            normalized.update(raw_state)

            #   
            if "score" in raw_state:
                normalized["score"] = raw_state["score"]
            elif "points" in raw_state:
                normalized["score"] = raw_state["points"]

            #    
            if "health" in raw_state:
                normalized["health"] = raw_state["health"]
            elif "lives" in raw_state:
                normalized["health"] = raw_state["lives"]

            #   
            if "position" in raw_state:
                normalized["position"] = raw_state["position"]
            elif "x" in raw_state and "y" in raw_state:
                normalized["position"] = (raw_state["x"], raw_state["y"])
            elif "player_x" in raw_state and "player_y" in raw_state:
                normalized["position"] = (raw_state["player_x"], raw_state["player_y"])

            #     
            if "game_over" in raw_state:
                normalized["game_over"] = raw_state["game_over"]
            elif "is_alive" in raw_state:
                normalized["game_over"] = not raw_state["is_alive"]
            elif "success" in raw_state:
                #   success True        （    ）
                if raw_state["success"]:
                    normalized["game_over"] = True

            #       
            if "game_won" in raw_state:
                normalized["game_won"] = raw_state["game_won"]
                #   success  ，      
                normalized["success"] = raw_state["game_won"]
                #           game_over True
                if raw_state["game_won"]:
                    normalized["game_over"] = True
            elif "success" in raw_state:
                normalized["success"] = raw_state["success"]
            elif "game_over" in normalized:
                #     game_over，    success
                normalized["success"] = not normalized["game_over"]
                #   game_over True game_won   ，  game_won False
                if normalized["game_over"] and "game_won" not in normalized:
                    normalized["game_won"] = False

            #   
            if "steps_taken" in raw_state:
                normalized["steps_taken"] = raw_state["steps_taken"]

            #     
            if "player_direction" in raw_state:
                normalized["player_direction"] = raw_state["player_direction"]

            #     
            if "distance_to_exit" in raw_state:
                normalized["distance_to_exit"] = raw_state["distance_to_exit"]

        return normalized
