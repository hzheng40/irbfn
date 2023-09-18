from copy import deepcopy
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.raster_feature_builder import (
    RasterFeatureBuilder,
)
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import (
    VectorMapFeatureBuilder,
)
from nuplan.planning.training.preprocessing.feature_preprocessor import (
    FeaturePreprocessor,
)
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.raster_utils import (
    get_agents_raster,
    get_baseline_paths_raster,
    get_ego_raster,
    get_roadmap_raster,
)
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)


def get_agents_trace_raster(
    ego_state: EgoState,
    agent_states: List[List[Agent]],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    raster_shape: Tuple[int, int],
    polygon_bit_shift: int = 9,
):
    """
    Based on nuplan.planning.training.preprocessing.features.raster_utils.get_agents_raster()
    """
    xmin, xmax = x_range
    ymin, ymax = y_range
    width, height = raster_shape

    alpha_max = 1.0
    alpha_min = 0.1

    agents_raster: npt.NDArray[np.float32] = np.zeros(raster_shape, dtype=np.float32)

    ego_to_global = ego_state.rear_axle.as_matrix()
    global_to_ego = np.linalg.inv(ego_to_global)
    north_aligned_transform = StateSE2(0, 0, np.pi / 2).as_matrix()

    # Retrieve the scenario's boxes.
    # tracked_objects = [deepcopy(tracked_object) for tracked_object in detections.tracked_objects]
    tracked_agents = [
        [deepcopy(tracked_object) for tracked_object in current_time_object]
        for current_time_object in agent_states
    ]

    # loop through time, fade color as time increase
    for ti in range(len(agent_states)):
        tracked_objects = tracked_agents[ti]
        # fade with time
        alpha = alpha_min + (alpha_max - alpha_min) * (
            (len(agent_states) - ti) / len(agent_states)
        )

        for tracked_object in tracked_objects:
            # Transform the box relative to agent.
            raster_object_matrix = (
                north_aligned_transform
                @ global_to_ego
                @ tracked_object.center.as_matrix()
            )
            raster_object_pose = StateSE2.from_matrix(raster_object_matrix)
            # Filter out boxes outside the raster.
            valid_x = x_range[0] < raster_object_pose.x < x_range[1]
            valid_y = y_range[0] < raster_object_pose.y < y_range[1]
            if not (valid_x and valid_y):
                continue

            # Get the 2D coordinate of the detected agents.
            raster_oriented_box = OrientedBox(
                raster_object_pose,
                tracked_object.box.length,
                tracked_object.box.width,
                tracked_object.box.height,
            )
            box_bottom_corners = raster_oriented_box.all_corners()
            x_corners = np.asarray([corner.x for corner in box_bottom_corners])  # type: ignore
            y_corners = np.asarray([corner.y for corner in box_bottom_corners])  # type: ignore

            # Discretize
            y_corners = (y_corners - ymin) / (ymax - ymin) * height  # type: ignore
            x_corners = (x_corners - xmin) / (xmax - xmin) * width  # type: ignore

            box_2d_coords = np.stack([x_corners, y_corners], axis=1)  # type: ignore
            box_2d_coords = np.expand_dims(box_2d_coords, axis=0)

            # Draw the box as a filled polygon on the raster layer.
            box_2d_coords = (box_2d_coords * 2**polygon_bit_shift).astype(np.int32)
            cv2.fillPoly(
                agents_raster,
                box_2d_coords,
                color=alpha * 1.0,
                shift=polygon_bit_shift,
                lineType=cv2.LINE_AA,
            )

    # Flip the agents_raster along the horizontal axis.
    agents_raster = np.asarray(agents_raster)
    agents_raster = np.flip(agents_raster, axis=0)
    agents_raster = np.ascontiguousarray(agents_raster, dtype=np.float32)

    return agents_raster


class RasterFeatureBuilderAllTrace(RasterFeatureBuilder):
    def __init__(
        self,
        map_features: Dict[str, int],
        num_input_channels: int,
        target_width: int,
        target_height: int,
        target_pixel_size: float,
        ego_width: float,
        ego_front_length: float,
        ego_rear_length: float,
        ego_longitudinal_offset: float,
        baseline_path_thickness: int,
        road_user_types: List[TrackedObjectType] = [
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
        ],
        vehicle_types: List[TrackedObjectType] = [TrackedObjectType.VEHICLE],
    ) -> None:
        super().__init__(
            map_features,
            num_input_channels,
            target_width,
            target_height,
            target_pixel_size,
            ego_width,
            ego_front_length,
            ego_rear_length,
            ego_longitudinal_offset,
            baseline_path_thickness,
        )

        self.vehicle_types = vehicle_types
        self.road_user_types = road_user_types

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> Raster:
        """Inherited, get all states instead of initial state"""
        history = current_input.history
        ego_states = history.ego_states
        observations = history.observations
        agent_vehicle_states = self._get_agent_traces_from_observations(
            observations, self.vehicle_types
        )
        agent_road_user_states = self._get_agent_traces_from_observations(
            observations, self.road_user_types
        )

        if isinstance(observations, List[DetectionsTracks]):
            return self._compute_feature(
                ego_states,
                agent_vehicle_states,
                agent_road_user_states,
                initialization.map_api,
            )
        else:
            raise TypeError(
                f"Observation was type {observations[0].detection_type()}. Expected DetectionsTracks"
            )

    def get_features_from_scenario(self, scenario: AbstractScenario) -> Raster:
        """Inherited, get all states instead of initial state"""
        ego_states = self._get_ego_traces(scenario)
        agent_vehicle_states = self._get_agent_traces(scenario, self.vehicle_types)
        agent_road_user_states = self._get_agent_traces(scenario, self.road_user_types)
        map_api = scenario.map_api

        return self._compute_feature(
            ego_states, agent_vehicle_states, agent_road_user_states, map_api
        )

    def _compute_feature(
        self,
        ego_states: List[EgoState],
        agent_vehicle_states: List[List[Agent]],
        agent_road_user_states: List[List[Agent]],
        map_api: AbstractMap,
    ) -> Raster:
        """Inherited, rasters full trace instead of initial"""
        # map layer
        roadmap_raster = get_roadmap_raster(
            ego_states[-1].agent,
            map_api,
            self.map_features,
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.target_pixel_size,
        )
        # ego layer
        ego_raster = get_ego_raster(
            self.raster_shape,
            self.ego_longitudinal_offset,
            self.ego_width_pixels,
            self.ego_front_length_pixels,
            self.ego_rear_length_pixels,
        )
        # baseline paths
        baseline_paths_raster = get_baseline_paths_raster(
            ego_states[-1].agent,
            map_api,
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.target_pixel_size,
            self.baseline_path_thickness,
        )
        # vehicle agent layer
        vehicle_agents_raster = get_agents_trace_raster(
            ego_states[-1],
            agent_vehicle_states,
            self.x_range,
            self.y_range,
            self.raster_shape,
        )
        # road user agent layer
        road_user_agents_raster = get_agents_trace_raster(
            ego_states[-1],
            agent_road_user_states,
            self.x_range,
            self.y_range,
            self.raster_shape,
        )

        collated_layers: npt.NDArray[np.float32] = np.dstack(
            [
                ego_raster,
                vehicle_agents_raster,
                road_user_agents_raster,
                roadmap_raster,
                baseline_paths_raster,
            ]
        ).astype(np.float32)

        # Ensures channel is the last dimension.
        if collated_layers.shape[-1] != self.num_input_channels:
            raise RuntimeError(
                f"Invalid raster numpy array. "
                f"Expected {self.num_input_channels} channels, got {collated_layers.shape[-1]} "
                f"Shape is {collated_layers.shape}"
            )

        return Raster(data=collated_layers)

    def _get_ego_traces(
        self, scenario: AbstractScenario, subsampling_steps: int = 10
    ) -> List[EgoState]:
        """Get all traces of ego states in scenario"""
        ego_traces = []
        for i in range(scenario.get_number_of_iterations()):
            ego_traces.append(scenario.get_ego_state_at_iteration(i))
        return ego_traces

    def _get_agent_traces(
        self,
        scenario: AbstractScenario,
        object_types: List[TrackedObjectType],
        subsampling_steps: int = 10,
    ) -> List[List[Agent]]:
        """Get all traces of agents in scenario by type"""
        agent_traces = []
        num_iter = scenario.get_number_of_iterations()
        try:
            subsample_idx = np.linspace(
                num_iter - 11,
                num_iter,
                num=subsampling_steps,
                endpoint=False,
                dtype=int,
            )
        except:
            subsample_idx = range(num_iter)
        for i in subsample_idx:
            agent_traces.append(
                scenario.get_tracked_objects_at_iteration(
                    i
                ).tracked_objects.get_tracked_objects_of_types(object_types)
            )
        return agent_traces

    def _get_agent_traces_from_observations(
        self,
        observations: List[Observation],
        object_types: List[TrackedObjectType],
    ) -> List[List[Agent]]:
        """Get all traces of agents in scenario by type"""
        agent_traces = []

        for i in range(len(observations)):
            agent_traces.append(
                observations[i].get_tracked_objects_of_types(object_types)
            )
        return agent_traces
