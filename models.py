# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

# """
# Data models for the First Rl Demo Environment.

# The first_rl_demo environment is a simple test environment that echoes back messages.
# """

# from openenv.core.env_server.types import Action, Observation
# from pydantic import Field


# class FirstRlDemoAction(Action):
#     """Action for the First Rl Demo environment - just a message to echo."""

#     message: str = Field(..., description="Message to echo back")


# class FirstRlDemoObservation(Observation):
#     """Observation from the First Rl Demo environment - the echoed message."""

#     echoed_message: str = Field(default="", description="The echoed message")
#     message_length: int = Field(default=0, description="Length of the echoed message")



# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Camera PTZ Preset Alignment Environment.

Defines the continuous action space for camera movement and the 
observation space for tracking alignment errors.
"""

from typing import List
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class CameraAction(Action):
    """
    Action for the Camera PTZ environment.
    
    Represents delta changes to the Pan, Tilt, and Zoom.
    Values are continuous floats typically in the range [-1, 1].
    """

    delta_pan: float = Field(
        ..., 
        ge=-1.0, 
        le=1.0, 
        description="Change in horizontal rotation"
    )
    delta_tilt: float = Field(
        ..., 
        ge=-1.0, 
        le=1.0, 
        description="Change in vertical rotation"
    )
    delta_zoom: float = Field(
        ..., 
        ge=-1.0, 
        le=1.0, 
        description="Change in focal length"
    )


class CameraObservation(Observation):
    """
    Observation from the Camera PTZ environment.
    
    Provides the agent with its current coordinates and the desired target coordinates.
    """

    current_ptz: List[float] = Field(
        default_factory=list, 
        description="Current [pan, tilt, zoom] values in range [-1, 1]"
    )
    target_ptz: List[float] = Field(
        default_factory=list, 
        description="Target [pan, tilt, zoom] preset to reach"
    )
    distance_to_target: float = Field(
        default=0.0, 
        description="Euclidean distance between current and target positions"
    )