import os
import time
import numpy as np

import pybullet as p
from surrol.tasks.psm_env import PsmEnv, goal_distance
from surrol.utils.pybullet_utils import (
    get_link_pose,
    wrap_angle,
    pairwise_link_collision,
    pairwise_collision
)
from surrol.const import ASSET_DIR_PATH


class MatchBoard(PsmEnv):
    POSE_BOARD = ((0.55, -0.02, 0.6771), (1.57, 0, 1.57))  # 0.675 + 0.011 + 0.001
    WORKSPACE_LIMITS1 = ((0.48, 0.60), (-0.05, 0.10), (0.676, 0.715))
    SCALING = 5.
    DISTANCE_THRESHOLD = 0.005
    OBJECT_LIBRARY = ['0', '1', '2', '4', '5', '6', 'A', 'B', 'C']

    def _env_setup(self):
        super(MatchBoard, self)._env_setup()
        self.has_object = True

        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1].mean(),
               workspace_limits[2][1])
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False

        # match board
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'match_board/match_board.urdf'),
                            np.array(self.POSE_BOARD[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_BOARD[1]),
                            useFixedBase=True,
                            globalScaling=0.25 * self.SCALING)
        self.obj_ids['fixed'].append(obj_id)  # 1
        self._block_size = get_link_pose(obj_id, 1)[0][1] - get_link_pose(obj_id, 0)[0][1]

        # objects
        self.obj = np.random.choice(self.OBJECT_LIBRARY)
        # while self.obj in ['1', 'A', '4']:
        #     self.obj = np.random.choice(self.OBJECT_LIBRARY)
        self.obj = 'C'
        #self.obj = 'C'
        # self.target_row = self.OBJECT_LIBRARY.index(self.obj) // 3
        # self.target_col = self.OBJECT_LIBRARY.index(self.obj) % 3
        self.target_row = np.random.randint(0, 3)
        self.target_col = np.random.randint(0, 3)
        pos, orn = get_link_pose(self.obj_ids['fixed'][-1], 2 * 6 + 2)
        pos = [pos[0] + 0.025 * self.SCALING + 0.01 * np.random.rand(), pos[1] + 0.03 * self.SCALING + 0.02 * np.random.rand(), pos[2]]
        file_name = f"letter_{self.obj}.urdf" if self.obj.isalpha() else f"number_{self.obj}.urdf"
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'match_board/' + file_name),
                                np.array(pos),
                                p.getQuaternionFromEuler((0, 0, 0)),
                                useFixedBase=False,
                                globalScaling=0.25 * self.SCALING)
        self.obj_ids['rigid'].append(obj_id)
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][-1], 0

        # render related setting
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-0.02 * self.SCALING, 0.015 * self.SCALING, 0.330 * self.SCALING),
            distance=0.87 * self.SCALING,
            yaw=90,
            pitch=-32,
            roll=0,
            upAxisIndex=2
        )

    def _get_obs(self) -> dict:
        robot_state = self._get_robot_state(idx=0)

        pos, _ = get_link_pose(self.obj_id, -1)
        object_pos = np.array(pos)
        pos, orn = get_link_pose(self.obj_id, self.obj_link1)
        waypoint_pos = np.array(pos)
        # rotations
        waypoint_rot = np.array(p.getEulerFromQuaternion(orn))
        # relative position state
        object_rel_pos = object_pos - robot_state[0: 3]
        # achieved_goal: obj_link0_pos + handle_pos
        lid_pos = np.array(get_link_pose(self.obj_ids['fixed'][-1], self.target_row * 6 + 3)[0])
        handle_pos = np.array(get_link_pose(self.obj_ids['fixed'][-1], self.target_row * 6 + 4)[0])
        achieved_goal = np.concatenate([object_pos, handle_pos])

        observation = np.concatenate([
            robot_state, object_pos.ravel(), object_rel_pos.ravel(),
            waypoint_pos.ravel(), waypoint_rot.ravel(), lid_pos, handle_pos, self.goal.copy()
        ])
        obs = {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy()
        }
        return obs

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        obj_goal = np.array(get_link_pose(self.obj_ids['fixed'][-1], self.target_row * 6 + self.target_col)[0])
        obj_goal[2] += 0.002 * self.SCALING
        handle_goal = np.array(get_link_pose(self.obj_ids['fixed'][-1], self.target_row * 6 + 4)[0])
        return np.concatenate([obj_goal, handle_goal]).copy()

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        p.resetBasePositionAndOrientation(self.obj_ids['fixed'][0], self.goal[:3], (0, 0, 0, 1))
        self._waypoints = []  # six waypoints
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        pos_handle = get_link_pose(self.obj_ids['fixed'][-1], self.target_row * 6 + 4)[0]
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
            else wrap_angle(orn[2] + np.pi)  # minimize the delta yaw
        yaw_handle = 0

        #-------------------------------Add noise-----------------------------------------
        noise_std = 0.02
        noise = np.clip(noise_std * np.random.random(3), -noise_std, noise_std)

        #----------------------------Subtask 1----------------------------
        self._waypoints.append(np.array([pos_handle[0], pos_handle[1] - 0.001 * self.SCALING,
                                       pos_handle[2] + 0.02 * self.SCALING, yaw_handle, -0.5]))  # above handle 0
        self._waypoints.append(np.array([pos_handle[0], pos_handle[1] - 0.001 * self.SCALING,
                                       pos_handle[2] + 0.01 * self.SCALING, yaw_handle, -0.5]))  # pre grasp 1
        self._waypoints.append(np.array([pos_handle[0], pos_handle[1] - 0.001 * self.SCALING,
                                       pos_handle[2] + 0.01 * self.SCALING, yaw_handle, -0.5]))  # grasp 2
        self._waypoints.append(np.array([pos_handle[0], pos_handle[1] + (2.85 + noise[1]) * self._block_size,
                                       pos_handle[2] + 0.01 * self.SCALING, yaw_handle, -0.5]))  # pull 3
        self._waypoints.append(np.array([pos_handle[0], pos_handle[1] + (2.85 + noise[1]) * self._block_size,
                                        pos_handle[2] + 0.03 * self.SCALING + noise[2], yaw_handle, -0.5]))  # above handle 4
        
        #----------------------------Subtask 2----------------------------                               
        self._waypoints.append(np.array([pos_obj[0], pos_obj[1],
                                        pos_obj[2] + 0.02 * self.SCALING, yaw, 0.5]))  # above object 5
        self._waypoints.append(np.array([pos_obj[0], pos_obj[1], 
                                        pos_obj[2], yaw, 0.5]))  # pre grasp 6  
        self._waypoints.append(np.array([pos_obj[0] , pos_obj[1], 
                                        pos_obj[2], yaw, -0.5]))  # grasp 7    
        self._waypoints.append(np.array([pos_obj[0], pos_obj[1], 
                                        pos_obj[2] + 0.03 * self.SCALING + noise[2], yaw, -0.5]))  # lift 8        

        #----------------------------Subtask 3----------------------------                                                      
        self._waypoints.append(np.array([self.goal[0], self.goal[1], 
                                        self.goal[2] + 0.03 * self.SCALING + noise[2], yaw, -0.5]))  # above goal 9  
        self._waypoints.append(np.array([self.goal[0], self.goal[1], 
                                        self.goal[2] + 0.03 * self.SCALING + noise[2], yaw, 0.5]))  # release 10

        #----------------------------Subtask 4----------------------------                                                                       
        self._waypoints.append(np.array([pos_handle[0], pos_handle[1] + 3.05 * self._block_size,
                                        pos_handle[2] + 0.03 * self.SCALING, yaw_handle, -0.5]))  # above handle 11
        self._waypoints.append(np.array([pos_handle[0], pos_handle[1] + 3.05 * self._block_size,
                                       pos_handle[2] + 0.01 * self.SCALING, yaw_handle, -0.5]))  # pre grasp 12
        self._waypoints.append(np.array([pos_handle[0], pos_handle[1] + 3.05 * self._block_size, 
                                       pos_handle[2] + 0.01 * self.SCALING, yaw_handle, -0.5]))  # grasp 13
        self._waypoints.append(np.array([pos_handle[0], pos_handle[1] + 0.003 * self.SCALING,
                                       pos_handle[2] + 0.01 * self.SCALING, yaw_handle, -0.5]))  # push 14 
        self._waypoints.append(np.array([pos_handle[0], pos_handle[1] + 0.003 * self.SCALING,
                                       pos_handle[2] + 0.02 * self.SCALING, yaw_handle, -0.5]))  # lift 15 
        self._waypoints_done = [False] * len(self._waypoints)

        #----------------------------Subgoals----------------------------
        self.subgoals = []
        self.subgoals.append(np.array([pos_handle[0], pos_handle[1] + (2.85 + noise[1]) * self._block_size, pos_handle[2] + 0.03 * self.SCALING + noise[2],
                                        pos_obj[0], pos_obj[1], pos_obj[2], 
                                        pos_handle[0], pos_handle[1] + (3 + noise[1]) * self._block_size, pos_handle[2]]))
        self.subgoals.append(np.array([pos_obj[0], pos_obj[1], pos_obj[2] + 0.03 * self.SCALING + noise[2],
                                        pos_obj[0], pos_obj[1], pos_obj[2] + 0.02 * self.SCALING + noise[2], 
                                        pos_handle[0], pos_handle[1] + (3 + noise[1]) * self._block_size, pos_handle[2]]))
        self.subgoals.append(np.array([self.goal[0], self.goal[1], self.goal[2] + 0.03 * self.SCALING + noise[2], 
                                        self.goal[0], self.goal[1], self.goal[2], 
                                        pos_handle[0], pos_handle[1] + (3 + noise[1]) * self._block_size, pos_handle[2]]))   
        self.subgoals.append(np.array([pos_handle[0], pos_handle[1] + 0.003 * self.SCALING, pos_handle[2] + 0.02 * self.SCALING,
                                        self.goal[0], self.goal[1], self.goal[2], 
                                        pos_handle[0], pos_handle[1], pos_handle[2]]))   

        self.pos_handle_target = pos_handle[1] + 3 * self._block_size
        self.pull_noise = noise_std * self._block_size

    # def _meet_contact_constraint_requirement(self):
    #     # add a contact constraint to the grasped block to make it stable
    #     #pose = get_link_pose(self.obj_id, -1)
    #     #return pose[0][2] > self.goal[2] + 0.01 * self.SCALING
    
    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # if self._waypoints_done[10] and not self._waypoints_done[11]:
        #     pos_handle = get_link_pose(self.obj_ids['fixed'][-1], self.target_row * 6 + 4)[0]
        #     for i in [11, 12, 13]:
        #         self._waypoints[i][1] = pos_handle[1]

        action = np.zeros(5)
        for i, waypoint in enumerate(self._waypoints):
            if self._waypoints_done[i]:
                continue
            delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
            delta_yaw = (waypoint[3] - obs['observation'][5]).clip(-0.4, 0.4)
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.75 if i not in [3, 14] else 0.475
            #scale_factor = 0.55
            delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 2e-3 and np.abs(delta_yaw) < np.deg2rad(5.):
                self._waypoints_done[i] = True
            break

        return action, i


if __name__ == "__main__":
    env = MatchBoard(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)