# first_rl_demo_environment.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# """
# First Rl Demo Environment Implementation.

# A simple test environment that echoes back messages sent to it.
# Perfect for testing HTTP server infrastructure.
# """

# from uuid import uuid4

# from openenv.core.env_server.interfaces import Environment
# from openenv.core.env_server.types import State

# try:
#     from ..models import FirstRlDemoAction, FirstRlDemoObservation
# except ImportError:
#     from models import FirstRlDemoAction, FirstRlDemoObservation


# class FirstRlDemoEnvironment(Environment):
#     """
#     A simple echo environment that echoes back messages.

#     This environment is designed for testing the HTTP server infrastructure.
#     It maintains minimal state and simply echoes back whatever message it receives.

#     Example:
#         >>> env = FirstRlDemoEnvironment()
#         >>> obs = env.reset()
#         >>> print(obs.echoed_message)  # "First Rl Demo environment ready!"
#         >>>
#         >>> obs = env.step(FirstRlDemoAction(message="Hello"))
#         >>> print(obs.echoed_message)  # "Hello"
#         >>> print(obs.message_length)  # 5
#     """

#     # Enable concurrent WebSocket sessions.
#     # Set to True if your environment isolates state between instances.
#     # When True, multiple WebSocket clients can connect simultaneously, each
#     # getting their own environment instance (when using factory mode in app.py).
#     SUPPORTS_CONCURRENT_SESSIONS: bool = True

#     def __init__(self):
#         """Initialize the first_rl_demo environment."""
#         self._state = State(episode_id=str(uuid4()), step_count=0)
#         self._reset_count = 0

#     def reset(self) -> FirstRlDemoObservation:
#         """
#         Reset the environment.

#         Returns:
#             FirstRlDemoObservation with a ready message
#         """
#         self._state = State(episode_id=str(uuid4()), step_count=0)
#         self._reset_count += 1

#         return FirstRlDemoObservation(
#             echoed_message="First Rl Demo environment ready!",
#             message_length=0,
#             done=False,
#             reward=0.0,
#         )

#     def step(self, action: FirstRlDemoAction) -> FirstRlDemoObservation:  # type: ignore[override]
#         """
#         Execute a step in the environment by echoing the message.

#         Args:
#             action: FirstRlDemoAction containing the message to echo

#         Returns:
#             FirstRlDemoObservation with the echoed message and its length
#         """
#         self._state.step_count += 1

#         message = action.message
#         length = len(message)

#         # Simple reward: longer messages get higher rewards
#         reward = length * 0.1

#         return FirstRlDemoObservation(
#             echoed_message=message,
#             message_length=length,
#             done=False,
#             reward=reward,
#             metadata={"original_message": message, "step": self._state.step_count},
#         )

#     @property
#     def state(self) -> State:
#         """
#         Get the current environment state.

#         Returns:
#             Current State with episode_id and step_count
#         """
#         return self._state




# preset_env.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Camera PTZ Preset Alignment Environment.

An RL environment where an agent must align a drifting PTZ camera 
to a specific target preset using continuous delta adjustments.
"""

import random
from uuid import uuid4
import numpy as np

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Assuming these models are defined in your models.py
try:
    from ..models import CameraAction, CameraObservation
except ImportError:
    from models import CameraAction, CameraObservation


class CameraPresetEnvironment(Environment):
    """
    Hospital PTZ Camera Alignment Environment.
    
    The agent receives the current [p, t, z] and target [p, t, z].
    It must output [dp, dt, dz] to minimize the alignment error.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    
    # Threshold for considering the camera "aligned"
    ALIGNMENT_THRESHOLD = 0.03
    MAX_STEPS = 100

    def __init__(self):
        """Initialize the camera environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.current_pos = np.zeros(3)  # [pan, tilt, zoom]
        self.target_pos = np.zeros(3)
        self.difficulty = "medium"

    def _get_random_coords(self):
        """Returns 3 floats in range [-1, 1]."""
        return np.array([random.uniform(-1, 1) for _ in range(3)])

    # def reset(self) -> CameraObservation:
    #     """
    #     Reset the camera to a drifted position and set a new target.
    #     """
    #     self._state = State(episode_id=str(uuid4()), step_count=0)
        
    #     # Initialize random target and a drifted current position
    #     self.target_pos = self._get_random_coords()
    #     self.current_pos = self._get_random_coords()

        
    def reset(self, task_id: str = "human_touch_medium", **kwargs) -> CameraObservation:
        # kwargs captures any extra arguments passed by the OpenEnv wrapper
        print(f"SERVER RECEIVED TASK: {task_id}") # You'll see this in the docker logs
        drift_scales = {
            "glitch_easy": 0.2, 
            "human_touch_medium": 0.5, 
            "hardware_drift_hard": 0.9
        }
        # Set the internal difficulty for metadata tracking
        self.difficulty = task_id 
        scale = drift_scales.get(task_id, 0.2)

        self.target_pos = self._get_random_coords()
        # Initial position is target + random noise scaled by difficulty
        noise = (np.random.rand(3) * 2 - 1) * scale
        self.current_pos = np.clip(self.target_pos + noise, -1.0, 1.0)
        return CameraObservation(
            current_ptz=self.current_pos.tolist(),
            target_ptz=self.target_pos.tolist(),
            distance_to_target=float(np.linalg.norm(self.current_pos - self.target_pos)),
            done=False,
            reward=0.0,
        )
    
    def step(self, action: CameraAction) -> CameraObservation:  # type: ignore[override]
        """
        Execute a step in the environment with improved progress-based rewards.
        """
        self._state.step_count += 1

        # 1. Store the previous error before moving
        previous_error = np.linalg.norm(self.current_pos - self.target_pos)

        # 2. Update Position
        # deltas = np.array([action.delta_pan, action.delta_tilt, action.delta_zoom])
        # Scale the deltas so the agent can only move 20% of the range per step
        MAX_VELOCITY = 0.2 
        deltas = np.array([action.delta_pan, action.delta_tilt, action.delta_zoom]) * MAX_VELOCITY

        # Apply deltas and clip to valid PTZ range [-1.0, 1.0]
        self.current_pos = np.clip(self.current_pos + deltas, -1.0, 1.0)

        # 3. Calculate New Distance (Error)
        current_error = np.linalg.norm(self.current_pos - self.target_pos)
        
        # 4. Calculate Progress Reward
        # reward > 0 if current_error < previous_error (moved closer)
        # reward < 0 if current_error > previous_error (moved further)
        reward = float(previous_error - current_error) * 10.0 

        # 5. Check Termination Conditions
        done = False
        
        # Success check: if within threshold, give a large bonus
        if current_error < self.ALIGNMENT_THRESHOLD:
            reward += 10.0  # Final completion bonus
            done = True
        # Timeout check: limit the episode length
        elif self._state.step_count >= self.MAX_STEPS:
            done = True

        # 6. Return standard Observation
        return CameraObservation(
            current_ptz=self.current_pos.tolist(),
            target_ptz=self.target_pos.tolist(),
            distance_to_target=float(current_error),
            done=done,
            reward=reward,
            metadata={
                "step": self._state.step_count,
                "difficulty": self.difficulty,
                "improvement": float(previous_error - current_error)
            },
        )
    

    @property
    def state(self) -> State:
        return self._state