#include "jaco_controller_integrated.h"

using namespace std;
using namespace jaco_controller_integrated;

JacoController::JacoController() : nh_(""), nh_local_("~")
{
  launch_args.push_back("jaco_controller_integrated");
  launch_args.push_back("move_group_only.launch");
  Poco::ProcessHandle ph_movegroupLaunch = Poco::Process::launch(
    "roslaunch",launch_args);
  ph_movegroup = new Poco::ProcessHandle(ph_movegroupLaunch);
  called = false;

  kill_args.push_back("kill");
  kill_args.push_back("/move_group");

  updateParams();

  printf(MOVEIT_CONSOLE_COLOR_BLUE "Move_group setup within controller.\n" MOVEIT_CONSOLE_COLOR_RESET);
  move_group = new moveit::planning_interface::MoveGroupInterface(PLANNING_GROUP_);

  execute_action_client_.reset(new actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction>(
      nh_, "j2n6s300/follow_joint_trajectory", false));

  printf(MOVEIT_CONSOLE_COLOR_BLUE "Move_group setup finished.\n" MOVEIT_CONSOLE_COLOR_RESET);

  joint_model_group = move_group->getCurrentState()->getJointModelGroup(PLANNING_GROUP_);
  move_group->setPlanningTime(0.2);
  printf(MOVEIT_CONSOLE_COLOR_BLUE "Joint states called\n" MOVEIT_CONSOLE_COLOR_RESET);
  debug = false;

  teleop_sub_ = nh_.subscribe("key_input", 10, &JacoController::teleopCallback, this);
  spacenav_sub_ = nh_.subscribe("spacenav/joy", 2, &JacoController::spacenavCallback, this);
  action_sub_ = nh_.subscribe("rl_action_output", 10, &JacoController::actionCallback, this);
  reset_sub_ = nh_.subscribe("reset_key", 10, &JacoController::resetCallback, this);
}

void JacoController::updateParams()
{
  nh_local_.param<float>("jaco_ros_controller/speed_constant", p_speed_constant, 0.01);
  nh_local_.param<bool>("jaco_ros_controller/cartesian", p_cartesian, false);
}

void JacoController::teleopCallback(const std_msgs::Int8::ConstPtr &msg)
{
  key_input = msg->data;
  //printf(MOVEIT_CONSOLE_COLOR_BLUE "Key In: %c\n", key_input);

  waypoints.clear();
  current_pose = move_group->getCurrentPose().pose;
  tf2::Quaternion q(current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w);
  tf2::Matrix3x3 m(q);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);

  if (debug)
  {
    ROS_INFO("pose_x: %f", current_pose.position.x);
    ROS_INFO("pose_y: %f", current_pose.position.y);
    ROS_INFO("pose_z: %f", current_pose.position.z);
    ROS_INFO("orientation_r: %f", roll);
    ROS_INFO("orientation_p: %f", pitch);
    ROS_INFO("orientation_y: %f", yaw);
  }


  waypoints.push_back(current_pose);
  target_pose = current_pose;

  switch (key_input)
  {
    case 'w':
      target_pose.position.y += 0.01;
      break;
    case 's':
      target_pose.position.y -= 0.01;
      break;
    case 'a':
      target_pose.position.x -= 0.01;
      break;
    case 'd':
      target_pose.position.x += 0.01;
      break;
    case 'e':
      target_pose.position.z += 0.01;
      break;
    case 'q':
      target_pose.position.z -= 0.01;
      break;

    case 'u':
      roll += 0.1;
      break;
    case 'j':
      roll -= 0.1;
      break;
    case 'h':
      pitch += 0.1;
      break;
    case 'k':
      pitch -= 0.1;
      break;
    case 'y':
      yaw += 0.1;
      break;
    case 'i':
      yaw -= 0.1;
      break;
    case '7':
      printf(MOVEIT_CONSOLE_COLOR_BLUE "Spacenav: True\n");
      spacenav_input = true;
      break;
    case '8':
      printf(MOVEIT_CONSOLE_COLOR_BLUE "Spacenav: False\n");
      spacenav_input = false;
      break;
    case '9':
      printf(MOVEIT_CONSOLE_COLOR_BLUE "Key: True\n");
      expert_input = true;
      break;
    case '0':
      printf(MOVEIT_CONSOLE_COLOR_BLUE "Key: False\n");
      expert_input = false;
      break;
  }

  if (debug)
  {
    ROS_INFO("target pose_x: %f", target_pose.position.x);
    ROS_INFO("target pose_y: %f", target_pose.position.y);
    ROS_INFO("target pose_z: %f", target_pose.position.z);
    ROS_INFO("target orientation_r: %f", roll);
    ROS_INFO("target orientation_p: %f", pitch);
    ROS_INFO("target orientation_y: %f", yaw);
  }
  
  moveit_msgs::RobotTrajectory trajectory;
  if (!p_cartesian && !expert_input)
  {
    tf2::Quaternion orientation;
    orientation.setRPY(roll, pitch, yaw);
    target_pose.orientation = tf2::toMsg(orientation);

    move_group->setPoseTarget(target_pose); //motion planning to a desired pose of the end-effector
    ROS_DEBUG_NAMED("","Planning Goal");
    move_group->plan(my_plan, 0.1);
    ROS_DEBUG_NAMED("","Planning Finished");

    trajectory = my_plan.trajectory_;
    control_msgs::FollowJointTrajectoryGoal goal;
    goal.trajectory = trajectory.joint_trajectory;
    ROS_DEBUG_NAMED("","Goal Sending");
    execute_action_client_->sendGoal(goal);
  }
  else if (p_cartesian && !expert_input)
  {
    waypoints.push_back(target_pose);
    fraction = move_group->computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);
    control_msgs::FollowJointTrajectoryGoal goal;
    goal.trajectory = trajectory.joint_trajectory;
    ROS_DEBUG_NAMED("","Goal Sending");
    execute_action_client_->sendGoal(goal);
  }
  else
  {
  }
}

void JacoController::spacenavCallback(const sensor_msgs::Joy::ConstPtr& msg)
{
  if (spacenav_input){
    waypoints.clear();
    current_pose = move_group->getCurrentPose().pose;
    tf2::Quaternion q(current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w);
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    if (debug)
    {
      ROS_INFO("pose_x: %f", current_pose.position.x);
      ROS_INFO("pose_y: %f", current_pose.position.y);
      ROS_INFO("pose_z: %f", current_pose.position.z);
      ROS_INFO("orientation_r: %f", roll);
      ROS_INFO("orientation_p: %f", pitch);
      ROS_INFO("orientation_y: %f", yaw);
    }

    waypoints.push_back(current_pose);
    target_pose = current_pose;
    target_pose.position.y += msg->axes[0] / 20.0;
    target_pose.position.x -= msg->axes[1] / 20.0;
    target_pose.position.z += msg->axes[2] / 20.0;
    roll += msg->axes[4] / 10.0;
    pitch -= msg->axes[3] / 10.0;
    yaw += msg->axes[5] / 10.0;

    if (debug)
    {
      ROS_INFO("target pose_x: %f", target_pose.position.x);
      ROS_INFO("target pose_y: %f", target_pose.position.y);
      ROS_INFO("target pose_z: %f", target_pose.position.z);
      ROS_INFO("target orientation_r: %f", roll);
      ROS_INFO("target orientation_p: %f", pitch);
      ROS_INFO("target orientation_y: %f", yaw);
    }

    moveit_msgs::RobotTrajectory trajectory;
    if (!p_cartesian)
    {
      ROS_DEBUG_NAMED("","Pose planning");
      tf2::Quaternion orientation;
      orientation.setRPY(roll, pitch, yaw);
      target_pose.orientation = tf2::toMsg(orientation);

      move_group->setPoseTarget(target_pose); //motion planning to a desired pose of the end-effector
      ROS_DEBUG_NAMED("","Planning Goal");
      move_group->plan(my_plan, 0.1);
      ROS_DEBUG_NAMED("","Planning Finished");

      trajectory = my_plan.trajectory_;
      control_msgs::FollowJointTrajectoryGoal goal;
      goal.trajectory = trajectory.joint_trajectory;
      ROS_DEBUG_NAMED("","Goal Sending");
      execute_action_client_->sendGoal(goal);
    }
    else if (p_cartesian)
    {
      ROS_DEBUG_NAMED("","Cartesian planning\n");
      waypoints.push_back(target_pose);
      fraction = move_group->computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);
      control_msgs::FollowJointTrajectoryGoal goal;
      goal.trajectory = trajectory.joint_trajectory;
      ROS_DEBUG_NAMED("","Goal Sending");
      execute_action_client_->sendGoal(goal);
    }
    else
    {
    }
  }
}

void JacoController::actionCallback(const std_msgs::Float32MultiArray &msg)
{
  printf(MOVEIT_CONSOLE_COLOR_BLUE "Action In: [");
  for (auto it = msg.data.begin(); it != msg.data.end(); it++)
  {
    printf("%.2f, ", *it);
  }
  printf("]\n");
  printf(MOVEIT_CONSOLE_COLOR_RESET);

  waypoints.clear();
  current_pose = move_group->getCurrentPose().pose;
  tf2::Quaternion q(current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w);
  tf2::Matrix3x3 m(q);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);

  if (debug)
  {
    ROS_INFO("pose_x: %f", current_pose.position.x);
    ROS_INFO("pose_y: %f", current_pose.position.y);
    ROS_INFO("pose_z: %f", current_pose.position.z);
    ROS_INFO("orientation_r: %f", roll);
    ROS_INFO("orientation_p: %f", pitch);
    ROS_INFO("orientation_y: %f", yaw);
  }

  waypoints.push_back(current_pose);
  target_pose = current_pose;
  target_pose.position.x += msg.data[0] / 100.0;
  target_pose.position.y += msg.data[1] / 100.0;
  target_pose.position.z += msg.data[2] / 100.0;
  roll += msg.data[3] / 20.0;
  pitch += msg.data[4] / 20.0;
  yaw += msg.data[5] / 20.0;

  if (debug)
  {
    ROS_INFO("target pose_x: %f", target_pose.position.x);
    ROS_INFO("target pose_y: %f", target_pose.position.y);
    ROS_INFO("target pose_z: %f", target_pose.position.z);
    ROS_INFO("target orientation_r: %f", roll);
    ROS_INFO("target orientation_p: %f", pitch);
    ROS_INFO("target orientation_y: %f", yaw);
  }

  moveit_msgs::RobotTrajectory trajectory;
  if (!p_cartesian)
  {
    ROS_DEBUG_NAMED("","Pose planning");
    tf2::Quaternion orientation;
    orientation.setRPY(roll, pitch, yaw);
    target_pose.orientation = tf2::toMsg(orientation);

    move_group->setPoseTarget(target_pose); //motion planning to a desired pose of the end-effector
    ROS_DEBUG_NAMED("","Planning Goal");
    move_group->plan(my_plan, 0.1);
    ROS_DEBUG_NAMED("","Planning Finished");

    trajectory = my_plan.trajectory_;
    control_msgs::FollowJointTrajectoryGoal goal;
    goal.trajectory = trajectory.joint_trajectory;
    ROS_DEBUG_NAMED("","Goal Sending");
    execute_action_client_->sendGoal(goal);
  }
  else if (p_cartesian)
  {
    ROS_DEBUG_NAMED("","Cartesian planning\n");
    waypoints.push_back(target_pose);
    fraction = move_group->computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);
    control_msgs::FollowJointTrajectoryGoal goal;
    goal.trajectory = trajectory.joint_trajectory;
    ROS_DEBUG_NAMED("","Goal Sending");
    execute_action_client_->sendGoal(goal);
  }
  else
  {
  }
}

void JacoController::resetCallback(const std_msgs::Int8::ConstPtr& msg)
{
  Poco::ProcessHandle ph_killLaunch = Poco::Process::launch("rosnode",kill_args);
  ph_kill = new Poco::ProcessHandle(ph_killLaunch);
  Poco::Process::wait(*ph_kill);
  free(ph_kill);
  Poco::Process::wait(*ph_movegroup);
  free(ph_movegroup);
  Poco::ProcessHandle ph_movegroupLaunch = Poco::Process::launch("roslaunch",launch_args);
  ph_movegroup = new Poco::ProcessHandle(ph_movegroupLaunch);
  reset_counter = 0;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "jaco_ros_controller");
  ros::AsyncSpinner spinner(2);
  spinner.start();
  JacoController jc;
  ros::waitForShutdown();

  return 0;
}