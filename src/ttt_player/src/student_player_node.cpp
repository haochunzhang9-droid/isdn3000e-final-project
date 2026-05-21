#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit_msgs/msg/robot_state.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <moveit_msgs/srv/get_position_ik.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include "ttt_interfaces/msg/game_snapshot.hpp"
#include "ttt_interfaces/msg/turn_plan.hpp"
#include "ttt_interfaces/msg/workspace_layout.hpp"
#include "ttt_interfaces/srv/plan_turn.hpp"
#include "ttt_interfaces/srv/register_player.hpp"

using namespace std::chrono_literals;

namespace {

std::vector<std::string> panda_joint_names() {
  return {"panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
          "panda_joint5", "panda_joint6", "panda_joint7"};
}

const std::vector<double> kHomePositions = {0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398};

// link8 orientation quaternion (x, y, z, w) for gripper pointing down
constexpr double kLink8QuatX = 0.9238795325112867;
constexpr double kLink8QuatY = -0.3826834323650898;
constexpr double kLink8QuatZ = 0.0;
constexpr double kLink8QuatW = 0.0;
// Fixed Z offset from panda_link8 to panda_hand_tcp
constexpr double kLink8TcpZOffset = 0.1034;

geometry_msgs::msg::Pose link8_pose_from_tcp_target(double x, double y, double z) {
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z + kLink8TcpZOffset;
  pose.orientation.x = kLink8QuatX;
  pose.orientation.y = kLink8QuatY;
  pose.orientation.z = kLink8QuatZ;
  pose.orientation.w = kLink8QuatW;
  return pose;
}

trajectory_msgs::msg::JointTrajectoryPoint make_point(
    const std::vector<double> &positions,
    double time_sec) {
  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.positions = positions;
  const auto whole_seconds = static_cast<int32_t>(std::floor(time_sec));
  point.time_from_start.sec = whole_seconds;
  point.time_from_start.nanosec =
      static_cast<uint32_t>((time_sec - static_cast<double>(whole_seconds)) * 1e9);
  return point;
}

moveit_msgs::msg::RobotTrajectory make_three_point_trajectory(
    const std::vector<double> &start_positions,
    const std::vector<double> &end_positions,
    double end_time_sec) {
  moveit_msgs::msg::RobotTrajectory trajectory;
  trajectory.joint_trajectory.joint_names = panda_joint_names();

  const std::vector<double> midpoint = [&]() {
    std::vector<double> result;
    result.reserve(start_positions.size());
    for (size_t index = 0; index < start_positions.size(); ++index) {
      result.push_back((start_positions[index] + end_positions[index]) * 0.5);
    }
    return result;
  }();

  trajectory.joint_trajectory.points.push_back(make_point(start_positions, 0.0));
  trajectory.joint_trajectory.points.push_back(make_point(midpoint, end_time_sec * 0.5));
  trajectory.joint_trajectory.points.push_back(make_point(end_positions, end_time_sec));
  return trajectory;
}

}  // namespace

class StudentPlayerNode : public rclcpp::Node {
 public:
  StudentPlayerNode() : Node("student_player") {
    this->declare_parameter<std::string>("player_name", this->get_name());
    this->declare_parameter<std::string>("plan_turn_service", "/student_player/plan_turn");

    player_name_ = this->get_parameter("player_name").as_string();
    plan_turn_service_ = this->get_parameter("plan_turn_service").as_string();

    cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    register_client_ =
        this->create_client<ttt_interfaces::srv::RegisterPlayer>("/ttt/register_player");

    ik_client_ = this->create_client<moveit_msgs::srv::GetPositionIK>(
        "/compute_ik",
        rmw_qos_profile_services_default,
        cb_group_);

    plan_turn_service_server_ = this->create_service<ttt_interfaces::srv::PlanTurn>(
        plan_turn_service_,
        std::bind(&StudentPlayerNode::handle_plan_turn, this, std::placeholders::_1,
                  std::placeholders::_2),
        rmw_qos_profile_services_default,
        cb_group_);

    register_timer_ =
        this->create_wall_timer(500ms, std::bind(&StudentPlayerNode::try_register, this));

    RCLCPP_INFO(this->get_logger(), "StudentPlayerNode initialized");
  }

 private:
  void try_register() {
    if (registered_ || registration_in_flight_) {
      return;
    }
    if (!register_client_->wait_for_service(100ms)) {
      return;
    }

    auto request = std::make_shared<ttt_interfaces::srv::RegisterPlayer::Request>();
    request->player_name = player_name_;
    request->service_name = plan_turn_service_;
    registration_in_flight_ = true;

    register_client_->async_send_request(
        request,
        [this](rclcpp::Client<ttt_interfaces::srv::RegisterPlayer>::SharedFuture future) {
          registration_in_flight_ = false;
          try {
            const auto response = future.get();
            if (!response->success) {
              RCLCPP_WARN(this->get_logger(), "Registration rejected: %s",
                          response->message.c_str());
              return;
            }
            registered_ = true;
            player_id_ = response->assigned_player_id;
            RCLCPP_INFO(this->get_logger(), "Registered as player_%u.", player_id_);
            register_timer_->cancel();
          } catch (const std::exception &exc) {
            RCLCPP_ERROR(this->get_logger(), "Registration failed: %s", exc.what());
          }
        });
  }

  // ================================================================
  // IK 实现（已完成）
  // ================================================================
  std::optional<std::vector<double>> compute_ik(
      const geometry_msgs::msg::Pose &target_pose,
      const std::vector<double> &seed_positions) {

    sensor_msgs::msg::JointState seed_state;
    seed_state.name = panda_joint_names();
    seed_state.position = seed_positions;

    auto request = std::make_shared<moveit_msgs::srv::GetPositionIK::Request>();
    request->ik_request.group_name = "panda_arm";
    request->ik_request.robot_state.joint_state = seed_state;
    request->ik_request.pose_stamped.header.frame_id = "panda_link0";
    request->ik_request.pose_stamped.pose = target_pose;
    request->ik_request.timeout.sec = 3;

    auto future = ik_client_->async_send_request(request);
    if (future.wait_for(4s) != std::future_status::ready) {
      RCLCPP_ERROR(this->get_logger(), "IK service timeout");
      return std::nullopt;
    }

    try {
      auto response = future.get();
      if (!response || response->error_code.val != 1) {
        RCLCPP_ERROR(this->get_logger(), "IK failed, error code = %d", 
                     response ? response->error_code.val : -1);
        return std::nullopt;
      }

      std::unordered_map<std::string, double> joint_map;
      for (size_t i = 0; i < response->solution.joint_state.name.size(); ++i) {
        joint_map[response->solution.joint_state.name[i]] = 
            response->solution.joint_state.position[i];
      }

      std::vector<double> result;
      for (const auto &name : panda_joint_names()) {
        result.push_back(joint_map.count(name) ? joint_map[name] : 0.0);
      }
      return result;
    } catch (...) {
      return std::nullopt;
    }
  }

  // ================================================================
  // find_piece_pose 实现（已完成）
  // ================================================================
  static geometry_msgs::msg::Pose find_piece_pose(
      const ttt_interfaces::msg::GameSnapshot &snapshot,
      uint8_t piece_id) {
    for (const auto &p : snapshot.pieces) {
      if (p.piece_id == piece_id && p.available) {
        return p.pose;
      }
    }
    // fallback
    geometry_msgs::msg::Pose fallback;
    fallback.orientation.w = 1.0;
    return fallback;
  }

  // ================================================================
  // handle_plan_turn 实现（已修复 3 个关键错误）
  // ================================================================
  void handle_plan_turn(
      const std::shared_ptr<ttt_interfaces::srv::PlanTurn::Request> request,
      std::shared_ptr<ttt_interfaces::srv::PlanTurn::Response> response) {

    RCLCPP_INFO(this->get_logger(), "=== [StudentPlayer] Received PlanTurn request ===");

    if (request->player_id != player_id_) {
      response->accepted = false;
      response->message = "Plan request does not match registered player id.";
      return;
    }

    // 1. 从 snapshot.pieces 选第一个可用棋子（owner == player_id_ && available）
    uint8_t piece_id = 255;
    for (const auto &p : request->snapshot.pieces) {
      if (p.available && p.owner == request->player_id) {
        piece_id = p.piece_id;
        break;
      }
    }
    if (piece_id == 255) {
      response->accepted = false;
      response->message = "No available pieces";
      return;
    }

    // 2. 从 legal_actions[0..8] 选第一个合法格子（正确使用 mask）
    uint8_t cell_id = 255;
    for (uint8_t c = 0; c < 9; ++c) {
      if (request->snapshot.legal_actions[c] == 1) {
        cell_id = c;
        break;
      }
    }
    if (cell_id == 255) {
      response->accepted = false;
      response->message = "No legal move";
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Selected move: piece %u -> cell %u", piece_id, cell_id);

    // 3. 计算 pick / place 目标
    auto pick_target = link8_pose_from_tcp_target(
        find_piece_pose(request->snapshot, piece_id).position.x,
        find_piece_pose(request->snapshot, piece_id).position.y,
        find_piece_pose(request->snapshot, piece_id).position.z);

    auto place_target = link8_pose_from_tcp_target(
        request->layout.cell_poses[cell_id].position.x,
        request->layout.cell_poses[cell_id].position.y,
        request->layout.cell_poses[cell_id].position.z);

    auto pick_j = compute_ik(pick_target, kHomePositions);
    auto place_j = compute_ik(place_target, kHomePositions);

    if (!pick_j || !place_j) {
      response->accepted = false;
      response->message = "IK failed";
      return;
    }

    RCLCPP_INFO(this->get_logger(), "IK solved successfully");

    // 4. 关键修复：必须填充 TurnPlan 前 3 个字段
    ttt_interfaces::msg::TurnPlan plan;
    plan.match_id    = request->match_id;
    plan.turn_index  = request->turn_index;
    plan.player_id   = request->player_id;
    plan.piece_id    = piece_id;
    plan.cell_id     = cell_id;

    plan.home_to_pick   = make_three_point_trajectory(kHomePositions, *pick_j, 1.5);
    plan.pick_to_home   = make_three_point_trajectory(*pick_j, kHomePositions, 1.5);
    plan.home_to_place  = make_three_point_trajectory(kHomePositions, *place_j, 1.5);
    plan.place_to_home  = make_three_point_trajectory(*place_j, kHomePositions, 1.5);

    response->accepted = true;
    response->message = "Valid TurnPlan from student player";
    response->plan = plan;

    RCLCPP_INFO(this->get_logger(), "=== [StudentPlayer] Plan returned successfully ===");
  }

  std::string player_name_;
  std::string plan_turn_service_;
  bool registered_{false};
  bool registration_in_flight_{false};
  uint8_t player_id_{255};

  rclcpp::CallbackGroup::SharedPtr cb_group_;
  rclcpp::Client<ttt_interfaces::srv::RegisterPlayer>::SharedPtr register_client_;
  rclcpp::Client<moveit_msgs::srv::GetPositionIK>::SharedPtr ik_client_;
  rclcpp::Service<ttt_interfaces::srv::PlanTurn>::SharedPtr plan_turn_service_server_;
  rclcpp::TimerBase::SharedPtr register_timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<StudentPlayerNode>();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}