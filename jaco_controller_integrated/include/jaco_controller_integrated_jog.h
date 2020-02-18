#include <ros/ros.h>
#include <chrono>
#include <algorithm>

#include <Eigen/Geometry>
//#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include "low_pass_filter.h"

#include <std_msgs/Int8.h>
#include <geometry_msgs/TwistStamped.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <control_msgs/FollowJointTrajectoryAction.h>

#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/macros/console_colors.h>


namespace jaco_controller_integrated_jog
{
class JacoController
{
public:
    JacoController();
private:
    void updateParams();
    void reset();
    void keyCallback(const std_msgs::Int8::ConstPtr& msg);
    void jointstateCallback(const sensor_msgs::JointState& msg);
    void moveJaco(const geometry_msgs::TwistStamped& cmd, const geometry_msgs::Pose& current_pose);
    bool addJointIncrements(sensor_msgs::JointState& output, const Eigen::VectorXd& increments) const;
    void lowPassFilterPositions(sensor_msgs::JointState& joint_state);
    void calculateJointVelocities(sensor_msgs::JointState& joint_state, const Eigen::ArrayXd& delta_theta);
    trajectory_msgs::JointTrajectory composeJointTrajMessage(sensor_msgs::JointState& joint_state) const;

    //ROS handles
    ros::NodeHandle nh_;
    ros::NodeHandle nh_local_;

    //Subscribers, Publishers
    ros::Subscriber teleop_sub_;
    ros::Subscriber joint_state_sub_;

    //Variables
    planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor_;
    std::unique_ptr<actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction> > execute_action_client_;
    robot_state::RobotStatePtr kinematic_state_;
    const robot_state::JointModelGroup* joint_model_group_;
    std::vector<LowPassFilter> position_filters_;
    
    sensor_msgs::JointState joint_state_;
    geometry_msgs::Pose current_pose;    
    moveit::planning_interface::MoveGroupInterface::Plan my_plan;
    
    Eigen::Isometry3d tf_moveit_to_cmd_frame_;
    Eigen::MatrixXd jacobian_, pseudo_inverse_, matrix_s_;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_;
    Eigen::ArrayXd delta_theta_;

    //Parameters
    const std::string PLANNING_GROUP_ = "arm";
    int key_input;
    float p_speed_constant;
    const float low_pass_filter_coeff = 2.0;
    const float linear_scale = 0.6;
    const float rotational_scale = 0.3;
    const float publish_period = 0.008;
};

}