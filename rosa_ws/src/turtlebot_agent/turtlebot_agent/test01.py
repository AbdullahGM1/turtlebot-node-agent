#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from math import cos, sin, sqrt
from std_srvs.srv import Empty
from turtlesim.msg import Pose
from turtlesim.srv import Spawn, TeleportAbsolute, TeleportRelative, Kill, SetPen
from langchain_ollama import ChatOllama
from langchain.agents import tool
from rosa import ROSA, RobotSystemPrompts

class TurtleBotAgentNode(Node):
    def __init__(self):
        super().__init__('turtlebot_agent_node')
        self.cmd_vel_pubs = {"turtle1": self.create_publisher(Twist, '/turtle1/cmd_vel', 10)}

        # Service clients
        self.spawn_client = self.create_client(Spawn, '/spawn')
        self.kill_client = self.create_client(Kill, '/kill')
        self.reset_client = self.create_client(Empty, '/reset')
        self.clear_client = self.create_client(Empty, '/clear')
        self.set_pen_clients = {}

        # Setup local LLM (Ollama)
        local_llm = ChatOllama(
            model="llama3.1:8b",
            temperature=0.0,
            max_retries=2,
            num_ctx=8192,
        )

# ============================= HELPER FUNCTIONS =============================
        def within_bounds(x: float, y: float) -> tuple:
            """Check if coordinates are within turtlesim bounds (0-11)"""
            if 0 <= x <= 11 and 0 <= y <= 11:
                return True, "Coordinates are within bounds."
            return False, f"({x}, {y}) will be out of bounds. Range is [0, 11]."

        def will_be_within_bounds(name: str, velocity: float, lateral: float, angle: float, duration: float = 1.0) -> tuple:
            """Predict if movement will keep turtle within simulator bounds"""
            pose = self.get_turtle_pose([name])
            current_x = pose[name].x
            current_y = pose[name].y
            current_theta = pose[name].theta

            if abs(angle) < 1e-6:
                new_x = current_x + (velocity * cos(current_theta) - lateral * sin(current_theta)) * duration
                new_y = current_y + (velocity * sin(current_theta) + lateral * cos(current_theta)) * duration
            else:
                radius = sqrt(velocity**2 + lateral**2) / abs(angle)
                center_x = current_x - radius * sin(current_theta)
                center_y = current_y + radius * cos(current_theta)
                angle_traveled = angle * duration
                new_x = center_x + radius * sin(current_theta + angle_traveled)
                new_y = center_y - radius * cos(current_theta + angle_traveled)

                for t in range(int(duration) + 1):
                    angle_t = current_theta + angle * t
                    x_t = center_x + radius * sin(angle_t)
                    y_t = center_y - radius * cos(angle_t)
                    in_bounds, _ = within_bounds(x_t, y_t)
                    if not in_bounds:
                        return False, f"Path goes out of bounds at ({x_t:.2f}, {y_t:.2f})"

            return within_bounds(new_x, new_y)

        
        # ============================= ROSA TOOLS (ROS 2) =============================

        @tool
        def spawn_turtle(name: str, x: float, y: float, theta: float) -> str:
            """Spawn a new turtle at specified coordinates"""
            in_bounds, message = within_bounds(x, y)
            if not in_bounds:
                return message

            name = name.replace("/", "")
            if self.spawn_client.wait_for_service(timeout_sec=5):
                req = Spawn.Request()
                req.x, req.y, req.theta, req.name = x, y, theta, name
                future = self.spawn_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)
                
                if future.result():
                    self.cmd_vel_pubs[name] = self.create_publisher(Twist, f'/{name}/cmd_vel', 10)
                    return f"{name} spawned at ({x}, {y}, θ={theta})"
                return f"Failed to spawn {name}: {future.exception()}"
            return "Spawn service unavailable"

        @tool
        def kill_turtle(names: list) -> str:
            """Remove specified turtles from the simulator"""
            response = ""
            for name in [n.replace("/", "") for n in names]:
                if self.kill_client.wait_for_service(timeout_sec=5):
                    req = Kill.Request()
                    req.name = name
                    future = self.kill_client.call_async(req)
                    rclpy.spin_until_future_complete(self, future)
                    if future.result():
                        self.cmd_vel_pubs.pop(name, None)
                        response += f"Killed {name}\n"
                    else:
                        response += f"Failed to kill {name}\n"
                else:
                    response += "Kill service unavailable\n"
            return response

        @tool
        def reset_turtlesim() -> str:
            """Reset the entire turtlesim environment"""
            if self.reset_client.wait_for_service(timeout_sec=5):
                future = self.reset_client.call_async(Empty.Request())
                rclpy.spin_until_future_complete(self, future)
                self.cmd_vel_pubs = {"turtle1": self.create_publisher(Twist, '/turtle1/cmd_vel', 10)}
                return "Turtlesim reset"
            return "Reset service unavailable"

        @tool
        def teleport_absolute(name: str, x: float, y: float, theta: float, hide_pen: bool = True) -> str:
            """Teleport turtle to absolute coordinates"""
            name = name.replace("/", "")
            client = self.create_client(TeleportAbsolute, f'/{name}/teleport_absolute')
            if client.wait_for_service(timeout_sec=5):
                req = TeleportAbsolute.Request()
                req.x, req.y, req.theta = x, y, theta
                future = client.call_async(req)
                rclpy.spin_until_future_complete(self, future)

                if hide_pen:
                    set_pen(name, 0, 0, 0, 1, 1)
                return f"{name} teleported to ({x}, {y}, θ={theta})"
            return f"Teleport service for {name} unavailable"

        @tool
        def teleport_relative(name: str, linear: float, angular: float) -> str:
            """Teleport turtle relative to current position"""
            name = name.replace("/", "")
            client = self.create_client(TeleportRelative, f'/{name}/teleport_relative')
            if client.wait_for_service(timeout_sec=5):
                req = TeleportRelative.Request()
                req.linear, req.angular = linear, angular
                future = client.call_async(req)
                rclpy.spin_until_future_complete(self, future)
                return f"{name} teleported relative ({linear}, {angular})"
            return f"Teleport service for {name} unavailable"

        @tool
        def set_pen(name: str, r: int, g: int, b: int, width: int, off: int) -> str:
            """Control the turtle's drawing pen properties"""
            name = name.replace("/", "")
            if name not in self.set_pen_clients:
                self.set_pen_clients[name] = self.create_client(SetPen, f'/{name}/set_pen')
            client = self.set_pen_clients[name]

            if client.wait_for_service(timeout_sec=5):
                req = SetPen.Request()
                req.r, req.g, req.b, req.width, req.off = r, g, b, width, off
                future = client.call_async(req)
                rclpy.spin_until_future_complete(self, future)
                return f"Pen set for {name}"
            return f"SetPen service for {name} unavailable"

        @tool
        def get_turtle_pose(names: list) -> dict:
            """Get current position/orientation of specified turtles"""
            poses = {}
            for name in [n.replace("/", "") for n in names]:
                try:
                    msg = self.wait_for_message(f'/{name}/pose', Pose, timeout_sec=5)
                    poses[name] = msg
                except Exception as e:
                    return {"Error": f"Failed to get pose for {name}: {e}"}
            return poses

        @tool
        def publish_twist_to_cmd_vel(name: str, velocity: float, lateral: float, angle: float, steps: int = 1) -> str:
            """Publish Twist commands for precise turtle movement"""
            name = name.replace("/", "")
            in_bounds, message = will_be_within_bounds(name, velocity, lateral, angle, steps)
            if not in_bounds:
                return message

            twist = Twist()
            twist.linear.x = velocity
            twist.linear.y = lateral
            twist.angular.z = angle

            pub = self.cmd_vel_pubs.get(name)
            if not pub:
                return f"No publisher found for {name}"

            for _ in range(steps):
                pub.publish(twist)
                rclpy.spin_once(self, timeout_sec=1.0)

            return f"Published Twist to {name} for {steps} steps"

        @tool
        def stop_turtle(name: str) -> str:
            """Stop turtle movement immediately"""
            return publish_twist_to_cmd_vel(name, 0.0, 0.0, 0.0, 1)

        @tool
        def has_moved_to_expected_coordinates(name: str, expected_x: float, expected_y: float, tolerance: float = 0.1) -> str:
            """Verify if turtle reached target coordinates"""
            pose = get_turtle_pose([name])
            current_x = pose[name].x
            current_y = pose[name].y
            distance = sqrt((current_x - expected_x)**2 + (current_y - expected_y)**2)
            if distance <= tolerance:
                return f"{name} reached target"
            return f"{name} missed target by {distance:.2f} units"

        # ============================= YOUR ORIGINAL TOOLS =============================
        @tool
        def move_forward(distance: float) -> str:
            """Move turtle forward for specified distance"""
            speed = 1.0
            duration = distance / speed

            twist = Twist()
            twist.linear.x = speed

            start_time = self.get_clock().now().nanoseconds
            while rclpy.ok():
                current_time = self.get_clock().now().nanoseconds
                elapsed = (current_time - start_time) / 1e9
                if elapsed >= duration:
                    break
                self.cmd_vel_pubs["turtle1"].publish(twist)
                rclpy.spin_once(self, timeout_sec=0.01)

            twist.linear.x = 0.0
            self.cmd_vel_pubs["turtle1"].publish(twist)
            return f"Moved forward {distance:.2f} units."

        @tool
        def rotate(degrees: float, direction: str = 'left') -> str:
            """Rotate turtle by specified degrees in given direction"""
            angular_speed = 1.0
            angle_radians = degrees * 3.141592653589793 / 180
            duration = angle_radians / angular_speed

            if direction.lower() not in ['left', 'right']:
                return "Invalid direction. Use 'left' or 'right'."

            twist = Twist()
            twist.angular.z = angular_speed if direction == 'left' else -angular_speed

            start_time = self.get_clock().now().nanoseconds
            while rclpy.ok():
                current_time = self.get_clock().now().nanoseconds
                elapsed = (current_time - start_time) / 1e9
                if elapsed >= duration:
                    break
                self.cmd_vel_pubs["turtle1"].publish(twist)
                rclpy.spin_once(self, timeout_sec=0.01)

            twist.angular.z = 0.0
            self.cmd_vel_pubs["turtle1"].publish(twist)
            return f"Rotated {degrees:.0f}° to the {direction}."

        # ============================= AGENT SETUP =============================
        prompts = RobotSystemPrompts(
            embodiment_and_persona="Advanced ROS 2 turtlesim controller",
            about_your_capabilities="Spawn/kill turtles, move, rotate, teleport, draw, verify positions",
            mission_and_objectives="Execute complex operations via natural language"
        )

        self.agent = ROSA(
            ros_version=2,
            llm=local_llm,
            tools=[
                spawn_turtle,
                kill_turtle,
                reset_turtlesim,
                teleport_absolute,
                teleport_relative,
                set_pen,
                get_turtle_pose,
                publish_twist_to_cmd_vel,
                stop_turtle,
                move_forward,
                rotate,
                has_moved_to_expected_coordinates
            ],
            prompts=prompts
        )

        self.get_logger().info("ROSA Agent ready. Type commands below:")

        # Interactive command loop
        while rclpy.ok():
            try:
                user_input = input("🧠 Your command > ")
                if user_input.strip().lower() in ["exit", "quit"]:
                    print("Exiting...")
                    break
                response = self.agent.invoke(user_input)
                print(f"🤖 ROSA: {response}")
            except KeyboardInterrupt:
                print("\n[!] Shutting down.")
                break

    def wait_for_message(self, topic: str, msg_type, timeout_sec: float):
        future = self.create_subscription(msg_type, topic, lambda msg: None, 10)
        if not self.wait_for_future(future, timeout_sec):
            raise Exception("Timeout waiting for message")
        return future.result()

def main(args=None):
    rclpy.init(args=args)
    node = TurtleBotAgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()