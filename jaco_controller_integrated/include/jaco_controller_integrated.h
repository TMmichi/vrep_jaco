#include <ros/ros.h>
#include <chrono>
#include <algorithm>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include "action_client/VrepInterface.hpp"
#include <std_msgs/Int8.h>
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
    void reset();
    void keyCallback(const std_msgs::Int8::ConstPtr& msg);

    //ROS handles
    ros::NodeHandle nh_;
    ros::NodeHandle nh_local_;

    //Subscribers, Publishers
    ros::Subscriber teleop_sub_;
    ros::Publisher key_check_pub_;

    //Variables
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
};

}