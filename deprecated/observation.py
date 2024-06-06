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
# Last Modified: 09/18/2023
# Observation wrappers

import jax
import jax.numpy as jnp
from flax import struct
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
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import *
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import (
    VectorMapFeatureBuilder,
)
from nuplan.planning.training.preprocessing.feature_preprocessor import (
    FeaturePreprocessor,
)
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import *
from nuplan.planning.training.preprocessing.utils.vector_preprocessing import (
    interpolate_points,
)

from nuplan_utils import RasterFeatureBuilderAllTrace


# TODO: this will need to be autodiff compatible
@struct.dataclass
class JaxVectorMap:
    coords: List[jax.Array]
    lane_groupings: List[List[jax.Array]]
    multi_scale_connections: List[Dict[int, FeatureDataType]]
    on_route_status: List[FeatureDataType]
    traffic_light_data: List[FeatureDataType]
    _lane_coord_dim: int = 2
    _on_route_status_encoding_dim: int = LaneOnRouteStatusData.encoding_dim()

    @classmethod
    def from_vector_map(cls, vector_map):
        # TODO: convert to jax types
        coords = None
        lane_groupings = None
        multi_scale_connections = None
        on_route_status = None
        traffic_light_data = None
        _lane_coord_dim = None
        _on_route_status_encoding_dim = None
        return cls(
            coords,
            lane_groupings,
            multi_scale_connections,
            on_route_status,
            traffic_light_data,
            _lane_coord_dim,
            _on_route_status_encoding_dim,
        )


class ObservationWrapper:
    def __init__(self):
        self.raster_feature_builder = RasterFeatureBuilderAllTrace(
            map_features={
                "LANE": 1.0,
                "INTERSECTION": 1.0,
            },
            num_input_channels=5,
            target_width=224,
            target_height=224,
            target_pixel_size=0.5,
            ego_width=2.297,
            ego_front_length=4.049,
            ego_rear_length=1.127,
            ego_longitudinal_offset=0.0,
            baseline_path_thickness=1,
        )
        self.vector_map_feature_builder = VectorMapFeatureBuilder(radius=20)

        self.ego_trajectory_target_builder = EgoTrajectoryTargetBuilder(
            TrajectorySampling(num_poses=10, time_horizon=5.0)
        )

        # TODO: might not need this
        # only works with scenarios
        # self.feature_preprocessor = FeaturePreprocessor(
        #     cache_path=None,
        #     feature_builders=[raster_feature_builder, vector_map_feature_builder],
        #     target_builders=[ego_trajectory_target_builder],
        #     force_feature_computation=False,
        # )

    def get_raster(
        self, planner_input: PlannerInput, planner_init: PlannerInitialization
    ) -> jax.Array:
        """
        Converts Planner input and planner initialization from nuPlan to JAX arrays representing multi-channel rasters
        Returns raster of shape [C x H x W]
        """
        rasters = self.raster_feature_builder.get_features_from_simulation(
            planner_input, planner_init
        ).data.T
        return jnp.asarray(rasters)

    def get_vector_map(
        self, planner_input: PlannerInput, planner_init: PlannerInitialization
    ) -> JaxVectorMap:
        """
        Converts Planner input and planner initialization from nuPlan to vector maps in JAX
        Returns raster of shape [C x H x W]

        TODO:
        """
        vector_map = self.vector_map_feature_builder.get_features_from_simulation(
            planner_input, planner_init
        )
        jax_vector_map = JaxVectorMap.from_vector_map(vector_map)
        return jax_vector_map
