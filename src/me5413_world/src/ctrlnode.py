#!/usr/bin/env python3

import math

import ipdb
import numpy as np
import rospy
import tf.transformations as tft
from dynamic_reconfigure.server import Server as DynServer
from geometry_msgs.msg import Pose, Twist, PoseArray
from nav_msgs.msg import Odometry, Path
from nmpc_controller import NMPCC

from me5413_world.cfg import path_publisherConfig


class PathTrackerNode1:
    def __init__(self):
        rospy.init_node("path_tracker_node")

        self.flag = True

        prob_params = {
            "control_dim": 2,
            "state_dim": 3,
            "max_vel": 10,
            "max_omega": 2,
            "init_pose": np.array([0, 0, 0]),
        }

        self.N = 20
        self.odom_world_robot_ = Odometry()
        self.mpc = NMPCC(
            T=0.2,
            N=self.N,
            prob_params=prob_params,
            Q=np.diag([5, 5, 0]),
            R=np.diag([5, 1]),
        )
        self.vel_ref = 5
        self.local_ref_path = None
        self.curr_pose = None
        self.global_path = None

        # ROS related
        self.sub_robot_odom = rospy.Subscriber("/gazebo/ground_truth/state", Odometry, self.robot_odom_callback)
        self.pub_cmd_vel = rospy.Publisher("/jackal_velocity_controller/cmd_vel", Twist, queue_size=1)
        # self.local_path_sub = rospy.Subscriber("/me5413_world/planning/local_path", Path, self.local_path_callback)
        self.global_path_sub = rospy.Subscriber("/me5413_world/planning/global_path", Path, self.global_path_callback)
        self.pred_pose_pub = rospy.Publisher("/me5413_world/planning/pred_pose", PoseArray, queue_size=1)
        self.local_ref_path_pub = rospy.Publisher("/me5413_world/planning/local_path", Path, queue_size=1)
        self.dyn_client = DynServer(path_publisherConfig, self.dyn_callback)

    def global_path_callback(self, msg: Path):
        self.global_path = msg

    def find_nearest_point(self):
        if self.global_path is None or self.curr_pose is None:
            return None
        min_dist = float("inf")
        for i, pose in enumerate(self.global_path.poses):
            dist = math.sqrt(
                (pose.pose.position.x - self.curr_pose[0]) ** 2 + (pose.pose.position.y - self.curr_pose[1]) ** 2
            )
            if dist < min_dist:
                min_dist = dist
                path_point_idx = i
        return path_point_idx

    def dyn_callback(self, config, level):
        self.vel_ref = config["speed_target"]
        rospy.loginfo("Reconfigure Request: speed_target={}".format(self.vel_ref))
        return config

    def pos_to_state(self, pos):
        yaw = self.quat_to_yaw(pos.orientation)
        state = np.array([pos.position.x, pos.position.y, yaw])
        return state

    def quat_to_yaw(self, quat):
        # yaw = np.arctan2(2 * (quat.w * quat.z + quat.x * quat.y), 1 - 2 * (quat.y**2 + quat.z**2))
        yaw = tft.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])[2]
        return yaw

    def robot_odom_callback(self, odom):
        if self.flag:
            self.flag = False
            pose = odom.pose.pose
            self.init_pose_ = self.pos_to_state(pose)
            return
        self.world_frame_ = odom.header.frame_id
        self.robot_frame_ = odom.child_frame_id
        self.odom_world_robot_ = odom
        self.curr_pose = [
            self.odom_world_robot_.pose.pose.position.x,
            self.odom_world_robot_.pose.pose.position.y,
            self.quat_to_yaw(self.odom_world_robot_.pose.pose.orientation),
        ]

    def local_path_callback(self, msg: Path):
        # self.pose_world_goal_ = path.pose[11].pose
        # self.pose_world_goal_ 应该包含后续一段时间内的路径规划信息
        # goal_poses = [pose.pose.position for pose in path.poses]
        self.local_ref_path = [
            [
                msg.poses[i].pose.position.x,
                msg.poses[i].pose.position.y,
                self.quat_to_yaw(msg.poses[i].pose.orientation),
            ]
            for i in range(11, min(len(msg.poses), 11 + self.N + 1))
        ]
        # if len(self.local_ref_path) < self.N + 1:  # extend the goal_poses to N+1
        #     self.local_ref_path += [self.local_ref_path[-1]] * (self.N + 1 - len(self.local_ref_path))

        if self.curr_pose is not None:
            for i in range(len(self.local_ref_path)):
                # if self.local_ref_path[i][2] < -math.pi:
                #     self.local_ref_path[i][2] += math.pi
                # if self.local_ref_path[i][2] > math.pi:
                #     self.local_ref_path[i][2] -= math.pi
                diff = self.local_ref_path[i][2] - self.curr_pose[2]
                while diff > math.pi:
                    diff -= 2 * math.pi
                    diff += 2 * math.pi
                self.local_ref_path[i][2] = self.curr_pose[2] + diff

    def compute_cmd_vel(self):
        if self.local_ref_path is None:
            return
        # rospy.loginfo(
        #     f"curr_x:{self.curr_pose[0]:.2f}, goal_x:{self.local_ref_path[0][0]:.2f}, curr_yaw: {self.curr_pose[2]:.3f}, goal_yaw:{self.local_ref_path[0][2]:.3f}, diff_yaw: {self.local_ref_path[0][2] - self.curr_pose[2]:.3f}"
        # )
        u = self.mpc.solve(self.curr_pose, self.local_ref_path, [[self.vel_ref, 0]] * self.N)
        x_pred = self.mpc.x_opti
        pose_array = PoseArray()
        pose_array.header.frame_id = self.world_frame_
        for i in range(self.N + 1):
            pose = Pose()
            pose.position.x = x_pred[i][0]
            pose.position.y = x_pred[i][1]
            quat = tft.quaternion_from_euler(0, 0, x_pred[i][2])
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            pose_array.poses.append(pose)
        self.pred_pose_pub.publish(pose_array)

        cmd_vel = Twist()
        cmd_vel.linear.x = u[0][0]
        cmd_vel.angular.z = u[0][1]
        return cmd_vel

    def get_local_ref_path(self):
        if self.global_path is None:
            return
        path_point_idx = self.find_nearest_point()
        self.local_ref_path = []
        path2pub = Path()
        for i in range(self.N + 1):
            if path_point_idx + i < len(self.global_path.poses):
                pose = [
                    self.global_path.poses[path_point_idx + i].pose.position.x,
                    self.global_path.poses[path_point_idx + i].pose.position.y,
                    self.quat_to_yaw(self.global_path.poses[path_point_idx + i].pose.orientation),
                ]
                path2pub.poses.append(self.global_path.poses[path_point_idx + i])
                self.local_ref_path.append(pose)
            else:
                j = path_point_idx + i - len(self.global_path.poses)
                pose = [
                    self.global_path.poses[j].pose.position.x,
                    self.global_path.poses[j].pose.position.y,
                    self.quat_to_yaw(self.global_path.poses[j].pose.orientation),
                ]
                path2pub.poses.append(self.global_path.poses[j])
                self.local_ref_path.append(pose)
        self.local_ref_path_pub.publish(path2pub)

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.get_local_ref_path()
            if self.local_ref_path is not None and self.curr_pose is not None:
                self.pub_cmd_vel.publish(self.compute_cmd_vel())
            rate.sleep()


if __name__ == "__main__":
    try:
        path_tracker_node = PathTrackerNode1()
        path_tracker_node.run()
    except rospy.ROSInterruptException:
        pass
