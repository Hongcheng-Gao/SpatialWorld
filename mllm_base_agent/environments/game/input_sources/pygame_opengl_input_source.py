"""
Pygame OpenGL        -   OpenGL  
"""
import os
import pygame
import numpy as np
from typing import Optional, Dict, Any
import time

# OpenGL    
from OpenGL.GL import *
from OpenGL.GLU import *

from core.input_source import GameInputSource
from core.data_classes import FrameData, GameState, Action


class PygameOpenGLInputSource(GameInputSource):
    """Pygame OpenGL      -   OpenGL  """

    def __init__(self):
        super().__init__()
        self.screen = None
        self.game_module = None
        self.frame_count = 0
        self.is_headless = False
        self.screen_size = (800, 600)  #       
        self.use_opengl = False

    def _get_capabilities(self) -> Dict[str, bool]:
        """  Pygame        """
        return {
            "real_time": True,
            "supports_actions": True,
            "supports_state_reading": True,
            "supports_reset": True,
            "headless": True,
            "opengl": True
        }

    def initialize(self, game_module=None, headless: bool = False,
                   screen_size: tuple = (800, 600), **kwargs) -> bool:
        """
           Pygame   -   OpenGL  

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
            has_display = os.environ.get('DISPLAY') is not None

            #         headless True    dummy  
            if headless and not has_display:
                os.environ["SDL_VIDEODRIVER"] = "dummy"
                os.environ["SDL_AUDIODRIVER"] = "dummy"
            else:
                #      ，    OpenGL
                try:
                    #        dummy    
                    if "SDL_VIDEODRIVER" in os.environ:
                        del os.environ["SDL_VIDEODRIVER"]
                except:
                    pass

            #    pygame
            pygame.init()

            #     OpenGL  
            try:
                from pygame.locals import DOUBLEBUF, OPENGL
                self.screen = pygame.display.set_mode(screen_size, DOUBLEBUF | OPENGL)
                self.use_opengl = True
                print("  OpenGL    ")
            except Exception as e:
                # OpenGL   ，      
                if headless:
                    self.screen = pygame.Surface(screen_size)
                else:
                    self.screen = pygame.display.set_mode(screen_size)
                    pygame.display.set_caption("MLLM Game Evaluation")
                print(f"OpenGL   ，      : {e}")

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
                if not self.use_opengl:
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

            #          -     OpenGL  
            if self.use_opengl:
                #  OpenGL   ，           
                #       
                width, height = self.screen_size

                #   OpenGL     
                glPixelStorei(GL_PACK_ALIGNMENT, 1)
                buffer = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

                #        numpy  
                image_array = np.frombuffer(buffer, dtype=np.uint8)
                image_array = image_array.reshape(height, width, 3)

                # OpenGL Y    ，    
                image_array = np.flipud(image_array)

            else:
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
                    "headless": self.is_headless,
                    "opengl": self.use_opengl
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
                if action.type == "key_press":
                    #              
                    if hasattr(self.game_module, 'execute_mapped_action'):
                        action_result = self._execute_mapped_action(action)
                    elif hasattr(self.game_module, 'get_action_mapping'):
                        #        ，       
                        try:
                            from core.action_mapping import ActionMapping
                            mapping = self.game_module.get_action_mapping(action.key)
                            if mapping and mapping.method_name:
                                method = getattr(self.game_module, mapping.method_name, None)
                                if method and callable(method):
                                    action_result = self._call_game_method(method, action)
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
                else:
                    action_result = False

                return action_result
            else:
                #       ，      
                if action.type == "key_press":
                    #       
                    key_event = pygame.event.Event(
                        pygame.KEYDOWN,
                        key=getattr(pygame, f"K_{action.key.upper()}", pygame.K_UNKNOWN)
                    )
                    pygame.event.post(key_event)
                    return True
                else:
                    return False

        except Exception as e:
            print(f"Failed to execute action: {e}")
            return False

    def _get_action_granularity(self, action: Action) -> Optional[str]:
        """Read small/medium/large movement granularity from action metadata."""
        if not action.metadata:
            return None

        granularity = action.metadata.get("granularity")
        if granularity is None and isinstance(action.metadata.get("json_data"), dict):
            granularity = action.metadata["json_data"].get("granularity")

        if granularity is None:
            return None

        value = str(granularity).strip().lower()
        return value if value in {"small", "medium", "large"} else None

    def _execute_mapped_action(self, action: Action) -> bool:
        return self._call_game_method(
            self.game_module.execute_mapped_action,
            action,
            include_key=True,
        )

    def _call_game_method(self, method, action: Action, include_key: bool = False) -> bool:
        granularity = self._get_action_granularity(action)
        args = [action.key] if include_key else []

        if granularity is not None:
            try:
                import inspect
                if "granularity" in inspect.signature(method).parameters:
                    return method(*args, granularity=granularity)
            except (TypeError, ValueError):
                pass

        return method(*args)

    def get_game_state(self) -> Optional[GameState]:
        """      """
        if self.game_module and hasattr(self.game_module, 'get_state'):
            try:
                state_dict = self.game_module.get_state()
                return GameState(
                    normalized_state=state_dict,
                    raw_state=state_dict
                )
            except Exception as e:
                print(f"Failed to get game state: {e}")
                return None
        return None

    def reset(self) -> bool:
        """    """
        if self.game_module and hasattr(self.game_module, 'reset'):
            try:
                self.game_module.reset()
                return True
            except Exception as e:
                print(f"Failed to reset game: {e}")
                return False
        return False

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

    def close(self):
        """     """
        pygame.quit()

    def _render_default_game(self):
        """      （           ）"""
        #        
        self.screen.fill((100, 100, 100))
        font = pygame.font.Font(None, 36)
        text = font.render("Default Game Render", True, (255, 255, 255))
        self.screen.blit(text, (50, 50))

    def get_info(self) -> Dict[str, Any]:
        """       """
        return {
            "type": "PygameOpenGLInputSource",
            "screen_size": self.screen_size,
            "headless": self.is_headless,
            "opengl": self.use_opengl,
            "frame_count": self.frame_count,
            "capabilities": self._get_capabilities()
        }
