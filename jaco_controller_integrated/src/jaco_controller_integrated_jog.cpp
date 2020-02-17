#include "jaco_controller_integrated_jog.h"


using namespace std;
using namespace jaco_controller_integrated_jog;

JacoController::JacoController() : nh_(""), nh_local_("~"){
  updateParams();
  teleop_sub_ = nh_.subscribe("key_input", 10, &JacoController::keyCallback, this);
  joint_state_sub_ = nh_.subscribe("j2n6s300/joint_states", 10, &JacoController::jointstateCallback, this);

  printf(MOVEIT_CONSOLE_COLOR_BLUE "Planning Scene Monitor Setup.\n" MOVEIT_CONSOLE_COLOR_RESET);
  planning_scene_monitor_ = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>("robot_description");
  if (!planning_scene_monitor_->getPlanningScene()){
    ROS_ERROR_STREAM_NAMED(LOGNAME, "Error in setting up the PlanningSceneMonitor.");
    exit(EXIT_FAILURE);
  }
  planning_scene_monitor_->startSceneMonitor();
  planning_scene_monitor_->startWorldGeometryMonitor(
      planning_scene_monitor::PlanningSceneMonitor::DEFAULT_COLLISION_OBJECT_TOPIC,
      planning_scene_monitor::PlanningSceneMonitor::DEFAULT_PLANNING_SCENE_WORLD_TOPIC,
      false /* skip octomap monitor */);
  planning_scene_monitor_->startStateMonitor();
  printf(MOVEIT_CONSOLE_COLOR_BLUE "Planning Scene Monitor Setup Finished.\n" MOVEIT_CONSOLE_COLOR_RESET);

  printf(MOVEIT_CONSOLE_COLOR_BLUE "Action Client Setup.\n" MOVEIT_CONSOLE_COLOR_RESET);
  execute_action_client_.reset(new actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction>(
        nh_, "j2n6s300/follow_joint_trajectory", false));
  printf(MOVEIT_CONSOLE_COLOR_BLUE "Action Client Setup Finished.\n" MOVEIT_CONSOLE_COLOR_RESET);

  const robot_model::RobotModelPtr& kinematic_model = planning_scene_monitor_->getRobotModelLoader();
  kinematic_state_ = std::make_shared<robot_state::RobotState>(kinematic_model);
  kinematic_state_->setToDefaultValues();
  joint_model_group_ = kinematic_model->getJointModelGroup(PLANNING_GROUP_);

  for (size_t i = 0; i < 6; ++i){
    position_filters_.emplace_back(low_pass_filter_coeff);
  }
}

void JacoController::updateParams(){
  nh_local_.param<float>("jaco_ros_controller/speed_constant", p_speed_constant, 0.01);
  //nh_local_.param<int>("jaco_ros_controller/iter", p_iter, 20);
}

void JacoController::reset(){
  execute_action_client_->cancelAllGoals();
}

void JacoController::keyCallback(const std_msgs::Int8::ConstPtr& msg){
  key_input = msg->data;
  printf(MOVEIT_CONSOLE_COLOR_BLUE "Key In: %c\n",key_input);
  printf(MOVEIT_CONSOLE_COLOR_RESET);

  geometry_msgs::TwistStamped cmd;
  switch(key_input){
    case 'w': cmd.twist.linear.y = 0.01; break;
    case 's': cmd.twist.linear.y = 0.01; break;
    case 'a': cmd.twist.linear.x = 0.01; break;
    case 'd': cmd.twist.linear.x = 0.01; break;
    case 'e': cmd.twist.linear.z = 0.01; break;
    case 'q': cmd.twist.linear.z = 0.01; break;

    case 'u': cmd.twist.angular.x += 0.1; i=1; break;
    case 'j': cmd.twist.angular.x -= 0.1; i=1; break;
    case 'h': cmd.twist.angular.y += 0.1; i=1; break;
    case 'k': cmd.twist.angular.y -= 0.1; i=1; break;
    case 'y': cmd.twist.angular.z += 0.1; i=1; break;
    case 'i': cmd.twist.angular.z -= 0.1; i=1; break;

    case 'r': this->reset(); break;
  }
  moveJaco(cmd, current_pose);
}

void JacoController::jointstateCallback(const sensor_msgs::JointState& msg){
  joint_state_ = msg;
}

void JacoController::moveJaco(const geometry_msgs::TwistStamped& cmd, const geometry_msgs::Pose& current_pose){
  kinematic_state_->setVariableValues(joint_state_);
  tf_moveit_to_cmd_frame_ = kinematic_state_->getGlobalLinkTransform(parameters_.planning_frame).inverse() *
                            kinematic_state_->getGlobalLinkTransform(parameters_.robot_link_command_frame);

  Eigen::Vector3d translation_vector(cmd.twist.linear.x, cmd.twist.linear.y, cmd.twist.linear.z);
  Eigen::Vector3d angular_vector(cmd.twist.angular.x, cmd.twist.angular.y, cmd.twist.angular.z);
  
  translation_vector = tf_moveit_to_cmd_frame_.linear() * translation_vector;
  angular_vector = tf_moveit_to_cmd_frame_.linear() * angular_vector;

  Eigen::VectorXd command(6);
  command[0] = translation_vector(0) * linear_scale * publish_period;
  command[1] = translation_vector(1) * linear_scale * publish_period;
  command[2] = translation_vector(2) * linear_scale * publish_period;
  command[3] = angular_vector(0) * rotational_scale* publish_period;
  command[4] = angular_vector(1) * rotational_scale * publish_period;
  command[5] = angular_vector(2) * rotational_scale * publish_period;

  // Convert from cartesian commands to joint commands
  jacobian_ = kinematic_state_->getJacobian(joint_model_group_);
  svd_ = Eigen::JacobiSVD<Eigen::MatrixXd>(jacobian_, Eigen::ComputeThinU | Eigen::ComputeThinV);
  matrix_s_ = svd_.singularValues().asDiagonal();
  pseudo_inverse_ = svd_.matrixV() * matrix_s_.inverse() * svd_.matrixU().transpose();
  delta_theta_ = pseudo_inverse_ * command;

  if (!addJointIncrements(joint_state_, delta_theta_))
    return false;

  lowPassFilterPositions(joint_state_);

  // Calculate joint velocities here so that positions are filtered and SRDF bounds still get checked
  calculateJointVelocities(joint_state_, delta_theta_);

  trajectory_msgs::JointTrajectory calc_traj;
  calc_traj = composeJointTrajMessage(joint_state_);

  //if (no_command)
  //  suddenHalt(calc_traj);

  control_msgs::FollowJointTrajectoryGoal goal;
  goal.trajectory = calc_traj;
  execute_action_client_->sendGoal(goal);
}

bool JacoController::addJointIncrements(sensor_msgs::JointState& output, const Eigen::VectorXd& increments) const {
  for (std::size_t i = 0, size = static_cast<std::size_t>(increments.size()); i < size; ++i){
    try{
      output.position[i] += increments[static_cast<long>(i)];
    }
    catch (const std::out_of_range& e){
      ROS_ERROR_STREAM_NAMED(LOGNAME, ros::this_node::getName() << " Lengths of output and increments do not match.");
      return false;
    }
  }
  return true;
}

void JacoController::lowPassFilterPositions(sensor_msgs::JointState& joint_state){
  for (size_t i = 0; i < position_filters_.size(); ++i){
    joint_state.position[i] = position_filters_[i].filter(joint_state.position[i]);
  }
}

void JacoController::calculateJointVelocities(sensor_msgs::JointState& joint_state, const Eigen::ArrayXd& delta_theta){
  for (int i = 0; i < delta_theta.size(); ++i){
    joint_state.velocity[i] = delta_theta[i] / publish_period;
  }
}

trajectory_msgs::JointTrajectory JacoController::composeJointTrajMessage(sensor_msgs::JointState& joint_state) const {
  trajectory_msgs::JointTrajectory new_joint_traj;
  new_joint_traj.header.frame_id = "world";
  new_joint_traj.header.stamp = ros::Time::now();
  new_joint_traj.joint_names = joint_state.name;

  trajectory_msgs::JointTrajectoryPoint point;
  point.time_from_start = ros::Duration(publish_period);
  if (true) //position traj
    point.positions = joint_state.position;
  if (false) //velocity traj
    point.velocities = joint_state.velocity;
  new_joint_traj.points.push_back(point);
  return new_joint_traj;
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