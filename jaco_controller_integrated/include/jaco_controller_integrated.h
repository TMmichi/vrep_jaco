#include <ros/ros.h>
#include <chrono>
#include <algorithm>
#include <Poco/Process.h>

#include <ros/network.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include "action_client/VrepInterface.hpp"
#include <std_msgs/Int8.h>
#include <std_msgs/Int8MultiArray.h>
#include <std_msgs/Float32MultiArray.h>
#include <sensor_msgs/Joy.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/macros/console_colors.h>


namespace jaco_controller_integrated
{

class JacoController
{
public:
    JacoController();
private:
    void updateParams();
    void teleopCallback(const std_msgs::Int8::ConstPtr& msg);
    void spacenavCallback(const sensor_msgs::Joy::ConstPtr& msg);
    void actionCallback(const std_msgs::Float32MultiArray& msg);
    void resetCallback(const std_msgs::Int8::ConstPtr& msg);

    //ROS handles
    ros::NodeHandle nh_;
    ros::NodeHandle nh_local_;

    //Subscribers, Publishers
    ros::Subscriber clock_sub_;
    ros::Subscriber teleop_sub_;
    ros::Subscriber spacenav_sub_;
    ros::Subscriber action_sub_;
    ros::Subscriber reset_sub_;
    ros::Publisher key_check_pub_;

    //Variables
    std::vector<std::string> launch_args;
    std::vector<std::string> kill_args;
    Poco::ProcessHandle* ph_movegroup;
    Poco::ProcessHandle* ph_kill;
    bool called;
    moveit::planning_interface::MoveGroupInterface* move_group;
    std::unique_ptr<actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction> > execute_action_client_;
    const robot_state::JointModelGroup* joint_model_group;
    std::vector<geometry_msgs::Pose> waypoints;
    geometry_msgs::Pose current_pose;
    geometry_msgs::Pose target_pose;
    moveit::planning_interface::MoveGroupInterface::Plan my_plan;

    //Parameters
    const std::string PLANNING_GROUP_ = "arm";
    int key_input;
    const double jump_threshold = 0.0;
    const double eef_step = 0.001;
    double fraction;
    float p_speed_constant;
    bool p_cartesian;
    bool debug;
    int reset_counter;
    bool expert_input = false;
};

}