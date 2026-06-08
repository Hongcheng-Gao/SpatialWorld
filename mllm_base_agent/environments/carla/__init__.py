"""
CARLA Environment Adapter
"""

from envs.carla.wrapper import CarlaEnvWrapper
from envs.carla.wrapper import  WalkerEnvWrapper

#      evaluate_action_sequence.py   import
VehicleExecutor = CarlaEnvWrapper
WalkerExecutor = WalkerEnvWrapper

__all__ = ["CarlaEnvWrapper", "WalkerEnvWrapper", "VehicleExecutor", "WalkerExecutor"]
