#include "jaco_controller_integrated.h"


using namespace std;
using namespace jaco_controller_integrated;

JacoController::JacoController() : nh_(""), nh_local_("~"){
  updateParams();

  teleop_sub_ = nh_.subscribe("key_input", 10, &JacoController::keyCallback, this);

  printf(MOVEIT_CONSOLE_COLOR_BLUE "Move_group setup within controller.\n" MOVEIT_CONSOLE_COLOR_RESET);
  move_group = new moveit::planning_interface::MoveGroupInterface(PLANNING_GROUP_);
  
  execute_action_client_.reset(new actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction>(
        nh_, "j2n6s300/follow_joint_trajectory", false));

  printf(MOVEIT_CONSOLE_COLOR_BLUE "Move_group setup finished.\n" MOVEIT_CONSOLE_COLOR_RESET);

  joint_model_group = move_group->getCurrentState()->getJointModelGroup(PLANNING_GROUP_);
  printf(MOVEIT_CONSOLE_COLOR_BLUE "Joint states called\n" MOVEIT_CONSOLE_COLOR_RESET);
}

void JacoController::updateParams(){
  nh_local_.param<float>("jaco_ros_controller/speed_constant", p_speed_constant, 0.01);
  nh_local_.param<bool>("jaco_ros_controller/cartesian", p_cartesian, false);
  cout << p_speed_constant << ", " << p_cartesian << endl;
}

void JacoController::reset(){
}

void JacoController::keyCallback(const std_msgs::Int8::ConstPtr& msg){
  key_input = msg->data;
  printf(MOVEIT_CONSOLE_COLOR_BLUE "Key In: %c\n",key_input);
  printf(MOVEIT_CONSOLE_COLOR_RESET);

  waypoints.clear();
  current_pose = move_group->getCurrentPose().pose;
  ROS_INFO("pose_x: %f",current_pose.position.x);
  ROS_INFO("pose_y: %f",current_pose.position.y);
  ROS_INFO("pose_z: %f",current_pose.position.z);

  tf2::Quaternion q(current_pose.orientation.x,current_pose.orientation.y,current_pose.orientation.z,current_pose.orientation.w);
  tf2::Matrix3x3 m(q);
  double roll,pitch,yaw;
  m.getRPY(roll,pitch,yaw);
  ROS_INFO("orientation_r: %f",roll);
  ROS_INFO("orientation_p: %f",pitch);
  ROS_INFO("orientation_y: %f",yaw);

  waypoints.push_back(current_pose);
  target_pose = current_pose;

  int i = 0;
  switch(key_input){
    case 'w': target_pose.position.y += 0.01;
    case 's': target_pose.position.y -= 0.01;
    case 'a': target_pose.position.x -= 0.01;
    case 'd': target_pose.position.x += 0.01;
    case 'e': target_pose.position.z += 0.01;
    case 'q': target_pose.position.z -= 0.01;

    case 'u': roll += 0.1; 
    case 'j': roll -= 0.1; 
    case 'h': pitch += 0.1;
    case 'k': pitch -= 0.1;
    case 'y': yaw += 0.1; 
    case 'i': yaw -= 0.1; i=1; break;
  }
  waypoints.push_back(target_pose);
  moveit_msgs::RobotTrajectory trajectory;
  cout << p_cartesian << endl;

  if(!p_cartesian && i==1){
    tf2::Quaternion orientation;
    orientation.setRPY(roll,pitch,yaw);
    target_pose.orientation = tf2::toMsg(orientation);

    move_group->setPoseTarget(target_pose); //motion planning to a desired pose of the end-effector
    ROS_INFO("Planning Goal");
    move_group->plan(my_plan);
    ROS_INFO("Planning Finished");

    trajectory = my_plan.trajectory_;
    control_msgs::FollowJointTrajectoryGoal goal;
    goal.trajectory = trajectory.joint_trajectory;
    ROS_INFO("Goal Sending");
    execute_action_client_->sendGoal(goal);
  }else if(p_cartesian && i==1){
    fraction = move_group->computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);
    control_msgs::FollowJointTrajectoryGoal goal;
    goal.trajectory = trajectory.joint_trajectory;
    ROS_INFO("Goal Sending");
    execute_action_client_->sendGoal(goal);
  }else{} 
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "jaco_ros_controller");
  ros::AsyncSpinner spinner(2);
  spinner.start();
  JacoController jc;
  ros::waitForShutdown();

  return 0;
}