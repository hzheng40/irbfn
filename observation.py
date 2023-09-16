# MIT License

# Copyright (c) 2023 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Author: Hongrui Zheng
# Last Modified: 09/11/2023
# Observation wrappers

import jax
import jax.numpy as jnp
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.geometry.torch_geometry import vector_set_coordinates_to_local_frame
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import *
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import *
from nuplan.planning.training.preprocessing.utils.vector_preprocessing import (
    interpolate_points,
)

# radius around ego to consider as observations
R = 50
# drivable map areas
DRIVABLE_MAP = [
    SemanticMapLayer.ROADBLOCK,
    SemanticMapLayer.ROADBLOCK_CONNECTOR,
    SemanticMapLayer.CARPARK_AREA,
]
# observation of other road participants
PARTICIPANTS = []
# map features to extract
MAP_FEATURES = ["LANE", "ROUTE_LANES", "CROSSWALK"]


def observation_to_vector_input(
    planner_input: PlannerInput, planner_init: PlannerInitialization
) -> jax.Array:
    """
    Converts planner input and planner initialization from nuPlan to JAX Arrays
    """
    pass


def rasterize_observation(
    planner_input: PlannerInput, planner_init: PlannerInitialization
) -> jax.Array:
    """
    Rasterize observations into multi-channel images
    """
    drivable_map_raster = planner_init.map_api.get_raster_map(DRIVABLE_MAP)


def get_drivable(map_api: AbstractMap, ego_state: EgoState, map_radius=R) -> jax.Array:
    """
    Get drivable areas from the map within radius around ego
    """
    position = ego_state.center.point
    drivable_area = map_api.get_proximal_map_objects(position, map_radius, DRIVABLE_MAP)


def get_agents(planner_input: PlannerInput, planner_init: PlannerInitialization):
    """
    """
    pass


def get_goal_route():
    """
    """
    pass