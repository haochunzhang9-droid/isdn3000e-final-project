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

constexpr double kLink8QuatX = 0.9238795325112867;
constexpr double kLink8QuatY = -0.3826834323650898;
constexpr double kLink8QuatZ = 0.0;
constexpr double kLink8QuatW = 0.0;
constexpr double kLink8TcpZOffset = 0.1034;

using Board = std::array<uint8_t, 9>;  // 0=empty, 1=player0, 2=player1 (student usually 1)

// ====================== TIC-TAC-TOE AI ======================
const std::vector<std::vector<int>> kWinLines = {
    {0,1,2},{3,4,5},{6,7,8},{0,3,6},{1,4,7},{2,5,8},{0,4,8},{2,4,6}};

int check_winner(const Board& b) {
  for (const auto& line : kWinLines) {
    if (b[line[0]] && b[line[0]] == b[line[1]] && b[line[0]] == b[line[2]]) return b[line[0]];
  }
  return 0;
}

bool is_full(const Board& b) { return std::none_of(b.begin(), b.end(), [](uint8_t x){return x==0;}); }

int minimax(Board board, bool is_max, uint8_t ai, int alpha=-1000, int beta=1000) {
  int w = check_winner(board);
  if (w == ai) return 10;
  if (w != 0) return -10;
  if (is_full(board)) return 0;

  if (is_max) {
    int best = -1000;
    for (int i=0; i<9; ++i) if (!board[i]) {
      board[i] = ai;
      best = std::max(best, minimax(board, false, ai, alpha, beta));
      board[i] = 0;
      alpha = std::max(alpha, best);
      if (beta <= alpha) break;
    }
    return best;
  } else {
    int best = 1000;
    uint8_t opp = (ai==1 ? 2 : 1);
    for (int i=0; i<9; ++i) if (!board[i]) {
      board[i] = opp;
      best = std::min(best, minimax(board, true, ai, alpha, beta));
      board[i] = 0;
      beta = std::min(beta, best);
      if (beta <= alpha) break;
    }
    return best;
  }
}

std::optional<uint8_t> get_best_move(const Board& board, uint8_t ai_player) {
  int best_score = -1000;
  std::optional<uint8_t> best = std::nullopt;
  for (uint8_t i = 0; i < 9; ++i) {
    if (board[i] == 0) {
      Board tmp = board; tmp[i] = ai_player;
      int score = minimax(tmp, false, ai_player);
      if (score > best_score) {
        best_score = score;
        best = i;
      }
    }
  }
  return best;
}

// ====================== ROS HELPERS ======================
geometry_msgs::msg::Pose link8_pose_from_tcp_target(double x, double y, double z) {
  geometry_msgs::msg::Pose p;
  p.position.x = x; p.position.y = y; p.position.z = z + kLink8TcpZOffset;
  p.orientation.x = kLink8QuatX; p.orientation.y = kLink8QuatY;
  p.orientation.z = kLink8QuatZ; p.orientation.w = kLink8QuatW;
  return p;
}

trajectory_msgs::msg::JointTrajectoryPoint make_point(const std::vector<double>& pos, double t) {
  trajectory_msgs::msg::JointTrajectoryPoint pt;
  pt.positions = pos;
  pt.time_from_start.sec = static_cast<int32_t>(std::floor(t));
  pt.time_from_start.nanosec = static_cast<uint32_t>((t - std::floor(t)) * 1e9);
  return pt;
}

moveit_msgs::msg::RobotTrajectory make_three_point_trajectory(
    const std::vector<double>& start, const std::vector<double>& end, double dur) {
  moveit_msgs::msg::RobotTrajectory traj;
  traj.joint_trajectory.joint_names = panda_joint_names();
  std::vector<double> mid(7);
  for (size_t i = 0; i < 7; ++i) mid[i] = (start[i] + end[i]) * 0.5;

  traj.joint_trajectory.points = {make_point(start, 0.0), make_point(mid, dur*0.5), make_point(end, dur)};
  return traj;
}

}  // namespace

class StudentPlayerNode : public rclcpp::Node {
 public:
  StudentPlayerNode() : Node("student_player") {
    declare_parameter("player_name", get_name());
    declare_parameter("plan_turn_service", "/student_player/plan_turn");

    player_name_ = get_parameter("player_name").as_string();
    plan_turn_service_ = get_parameter("plan_turn_service").as_string();

    auto cb_group = create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    register_client_ = create_client<ttt_interfaces::srv::RegisterPlayer>("/ttt/register_player");
    ik_client_ = create_client<moveit_msgs::srv::GetPositionIK>("/compute_ik", rmw_qos_profile_services_default, cb_group);

    plan_turn_service_server_ = create_service<ttt_interfaces::srv::PlanTurn>(
        plan_turn_service_,
        std::bind(&StudentPlayerNode::handle_plan_turn, this, std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default, cb_group);

    register_timer_ = create_wall_timer(500ms, std::bind(&StudentPlayerNode::try_register, this));
  }

 private:
  // 成员变量声明（必须在使用前定义）
  std::string player_name_, plan_turn_service_;
  bool registered_{false}, registration_in_flight_{false};
  uint8_t player_id_{255};

  rclcpp::Client<ttt_interfaces::srv::RegisterPlayer>::SharedPtr register_client_;
  rclcpp::Client<moveit_msgs::srv::GetPositionIK>::SharedPtr ik_client_;
  rclcpp::Service<ttt_interfaces::srv::PlanTurn>::SharedPtr plan_turn_service_server_;
  rclcpp::TimerBase::SharedPtr register_timer_;

  void try_register() {
    if (registered_ || registration_in_flight_) return;
    if (!register_client_->wait_for_service(100ms)) return;

    auto req = std::make_shared<ttt_interfaces::srv::RegisterPlayer::Request>();
    req->player_name = player_name_;
    req->service_name = plan_turn_service_;
    registration_in_flight_ = true;

    // 修正异步调用的future类型和lambda捕获
    using FutureType = rclcpp::Client<ttt_interfaces::srv::RegisterPlayer>::SharedFuture;
    register_client_->async_send_request(req, [this](FutureType future) {
      registration_in_flight_ = false;
      try {
        auto res = future.get();
        if (res->success) {
          registered_ = true;
          player_id_ = res->assigned_player_id;
          RCLCPP_INFO(get_logger(), "✅ Registered as player_%u", player_id_);
          register_timer_->cancel();
        } else {
          RCLCPP_WARN(get_logger(), "Registration rejected: %s", res->message.c_str());
        }
      } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Registration exception: %s", e.what());
      }
    });
  }

  std::optional<std::vector<double>> compute_ik(const geometry_msgs::msg::Pose& target, const std::vector<double>& seed) {
    sensor_msgs::msg::JointState seed_state;
    seed_state.name = panda_joint_names();
    seed_state.position = seed;

    auto req = std::make_shared<moveit_msgs::srv::GetPositionIK::Request>();
    req->ik_request.group_name = "panda_arm";
    req->ik_request.robot_state.joint_state = seed_state;
    req->ik_request.pose_stamped.header.frame_id = "panda_link0";
    req->ik_request.pose_stamped.pose = target;
    req->ik_request.timeout.sec = 3;

    auto future = ik_client_->async_send_request(req);
    if (future.wait_for(4s) != std::future_status::ready) return std::nullopt;

    try {
      auto res = future.get();
      if (!res || res->error_code.val != 1) return std::nullopt;

      std::unordered_map<std::string, double> m;
      for (size_t i = 0; i < res->solution.joint_state.name.size(); ++i)
        m[res->solution.joint_state.name[i]] = res->solution.joint_state.position[i];

      std::vector<double> out;
      for (const auto& n : panda_joint_names())
        out.push_back(m.count(n) ? m[n] : 0.0);
      return out;
    } catch (...) { return std::nullopt; }
  }

  static geometry_msgs::msg::Pose find_piece_pose(const ttt_interfaces::msg::GameSnapshot& snap, uint8_t pid) {
    for (const auto& p : snap.pieces)
      if (p.piece_id == pid && p.available) return p.pose;
    geometry_msgs::msg::Pose fallback;
    fallback.orientation.w = 1.0;
    return fallback;
  }

  // 构建 Board（从 pieces 推断当前局面）
  Board build_board(const ttt_interfaces::msg::GameSnapshot& snap) {
    Board b{};
    for (const auto& p : snap.pieces) {
      if (!p.available && p.owner != 0) {  // 已放置的棋子
        if (p.cell_id < 9) b[p.cell_id] = p.owner;
      }
    }
    return b;
  }

  void handle_plan_turn(
      const std::shared_ptr<ttt_interfaces::srv::PlanTurn::Request> req,
      std::shared_ptr<ttt_interfaces::srv::PlanTurn::Response> res) {

    if (req->player_id != player_id_) {
      res->accepted = false; res->message = "ID mismatch"; return;
    }

    Board board = build_board(req->snapshot);

    std::vector<uint8_t> my_pieces;
    for (const auto& p : req->snapshot.pieces) {
      if (p.available && p.owner == req->player_id) my_pieces.push_back(p.piece_id);
    }
    if (my_pieces.empty()) {
      res->accepted = false; res->message = "No available pieces"; return;
    }

    auto best_cell = get_best_move(board, req->player_id);
    if (!best_cell.has_value()) {
      for (uint8_t c = 0; c < 9; ++c) {
        if (req->snapshot.legal_actions[c] == 1) { best_cell = c; break; }
      }
    }
    if (!best_cell.has_value()) {
      res->accepted = false; res->message = "No legal move"; return;
    }

    uint8_t piece_id = my_pieces[0];
    uint8_t cell_id = *best_cell;

    auto pick_target = link8_pose_from_tcp_target(
        find_piece_pose(req->snapshot, piece_id).position.x,
        find_piece_pose(req->snapshot, piece_id).position.y,
        find_piece_pose(req->snapshot, piece_id).position.z);

    auto place_target = link8_pose_from_tcp_target(
        req->layout.cell_poses[cell_id].position.x,
        req->layout.cell_poses[cell_id].position.y,
        req->layout.cell_poses[cell_id].position.z);

    auto pick_j = compute_ik(pick_target, kHomePositions);
    auto place_j = compute_ik(place_target, kHomePositions);

    if (!pick_j || !place_j) {
      res->accepted = false; res->message = "IK failed"; return;
    }

    RCLCPP_INFO(get_logger(), "AI: piece %u -> cell %u", piece_id, cell_id);

    ttt_interfaces::msg::TurnPlan plan;
    plan.match_id = req->match_id;
    plan.turn_index = req->turn_index;
    plan.player_id = req->player_id;
    plan.piece_id = piece_id;
    plan.cell_id = cell_id;

    plan.home_to_pick   = make_three_point_trajectory(kHomePositions, *pick_j, 1.5);
    plan.pick_to_home   = make_three_point_trajectory(*pick_j, kHomePositions, 1.5);
    plan.home_to_place  = make_three_point_trajectory(kHomePositions, *place_j, 1.5);
    plan.place_to_home  = make_three_point_trajectory(*place_j, kHomePositions, 1.5);

    res->accepted = true;
    res->message = "Minimax AI Ready";
    res->plan = plan;
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<StudentPlayerNode>();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}