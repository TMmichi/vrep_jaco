/** @file VrepInterface.hpp
 * Acts as the interface between ROS and V-REP for controlling the Jaco arm.
 * Implements ROS's control_msgs::FollowJointTrajectoryAction action interface,
 * and uses V-REP's C++ api as this is faster than V-REP's ROS api.
 */
#ifndef JACO_CONTROL_VREP_INTERFACE_HPP_
#define JACO_CONTROL_VREP_INTERFACE_HPP_

#include <unordered_map>
#include <string>
#include <iostream>
#include <queue>

#include <ros/ros.h>
#include <ros/console.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/JointState.h>
#include <actionlib/server/simple_action_server.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <geometry_msgs/Pose.h>

class VrepInterface {

public:
    VrepInterface(ros::NodeHandle& n);
    void publishWorker(const ros::WallTimerEvent& e);
    void trajCB(const control_msgs::FollowJointTrajectoryGoalConstPtr &goal);

private:
    bool initJoints(std::string inPrefix, std::string outPrefix, int numJoints,
            sensor_msgs::JointState& jointState, std::vector<int>& jointHandles,
            int suffixCode = -1);
    std::vector<double> interpolate( const std::vector<double>& last,
            const std::vector<double>& current, double alpha);
    void updateJointState();
    void publishJointInfo();
    std::vector<double> getVrepPosition();
    void setVrepPosition(const std::vector<double>& pos);
    
    void getKey(const )

    /** ROS actionlib server for trajectory commands */
    std::unique_ptr<actionlib::SimpleActionServer<control_msgs::FollowJointTrajectoryAction>> trajAS_;

    /** VREP connection id */
    int clientID_;
    /** VREP handles of arm joints */
    std::vector<int> jointHandles_;

    /** Publisher for transformed current joint states */
    ros::Publisher jointPub_;
    /** Publisher for feedback states */
    ros::Publisher feedbackPub_;
    /** Subscriber to target torques */
    ros::Subscriber torqueSub_;

    /** Stores joint state */
    sensor_msgs::JointState jointState_;
    control_msgs::FollowJointTrajectoryFeedback feedback_;
    /** Target torques (torque mode) */
    std::vector<double> targetTorques_;

    /** Number of joints in jaco arm = 6 */
    int numArmJoints_;
    /** Number of finger joints = 3 */
    int numFingerJoints_;
    /** Number of finger tip joints = 3 */
    int numFingerTipJoints_;
    /** Rate for publishing joint info in Hz */
    double feedbackRate_;
    /** Rate for setting joint position commands */
    ros::Rate posUpdateRate_;

    /** Timer for publishWorker */
    ros::WallTimer publishWorkerTimer_;

    /** Torque mode for robot control. */
    bool torqueMode_;
    /** Whether to use synchronous mode with V-REP */
    bool sync_;
};

#endif /* JACO_CONTROL_VREP_INTERFACE_HPP_ */
