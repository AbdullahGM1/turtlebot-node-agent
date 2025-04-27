#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
# from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.agents import tool
from rosa import ROSA, RobotSystemPrompts
import math

# Load environment variables
#load_dotenv()

class TurtleBotAgentNode(Node):
    def __init__(self):
        super().__init__('turtlebot_agent_node')
        self.publisher_ = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)


# ============================= Setup local LLM (Ollama) =============================

        local_llm = ChatOllama(
            model="llama3.1:8b",
            temperature=0.0,
            max_retries=2,
            num_ctx=8192,
        )

# ============================= ROSA TOOLS (ROS 2) =============================
        @tool
        def publish_linear_motion(distance: float) -> str:
            """
            Move the TurtleSim turtle forward/backward by the specified distance (in turtlesim units).

            """
            linear_speed = 1.0  # units/second
            
            twist = Twist()
      
            if distance < 0:
                duration = abs(distance) / linear_speed
                twist.linear.x = -linear_speed
            else:
                duration = distance / linear_speed
                twist.linear.x = linear_speed
            
                
            start_time = self.get_clock().now().nanoseconds
            while rclpy.ok():
                current_time = self.get_clock().now().nanoseconds
                elapsed = (current_time - start_time) / 1e9  # Convert ns to seconds
                if elapsed >= duration:
                    break
                self.publisher_.publish(twist)
                rclpy.spin_once(self, timeout_sec=0.01)

            twist.linear.x = 0.0
            self.publisher_.publish(twist)
            return f"Moved forward {distance:.2f}."

        @tool
        def publish_angular_motion(angle: float) -> str:
            """
            Rotate the TurtleSim turtle by specified degrees in the given direction (left/right).
            """
            angular_speed = 1.0  # radians/second
            angle_radians = math.radians(angle)

            twist = Twist()

            if angle_radians < 0:
                duration = abs(angle_radians) / angular_speed
                twist.angular.z = -angular_speed
                direction = "left"
            else:
                duration = angle_radians / angular_speed
                twist.angular.z = angular_speed
                direction = "right"

            start_time = self.get_clock().now().nanoseconds
            while rclpy.ok():
                current_time = self.get_clock().now().nanoseconds
                elapsed = (current_time - start_time) / 1e9
                if elapsed >= duration:
                    break
                self.publisher_.publish(twist)
                rclpy.spin_once(self, timeout_sec=0.01)

            twist.angular.z = 0.0
            self.publisher_.publish(twist)
            return f"Rotated {angle_radians:.0f}° to the {direction}."
        
# ============================= AGENT SETUP =============================
        # Prompts
        prompts = RobotSystemPrompts(
            embodiment_and_persona="You are a smart TurtleBot in turtlesim.",
            about_your_capabilities="You can move forward and rotate left/right using degrees.",
            mission_and_objectives="Help users control the turtle with precise natural language commands."
        )

        # Initialize ROSA
        self.agent = ROSA(
            ros_version=2,
            llm=local_llm,
            tools=[publish_linear_motion, publish_angular_motion],
            prompts=prompts
        )

        self.get_logger().info("ROSA TurtleBot Agent is ready. Type a command:")

# ============================= Interactive command loop =============================

        while rclpy.ok():
            try:
                user_input = input("🧠 Your command > ")
                if user_input.strip().lower() in ["exit", "quit"]:
                    print("Exiting...")
                    break
                response = self.agent.invoke(user_input)
                print(f"🤖 ROSA: {response}")
            except KeyboardInterrupt:
                print("\n[!] Interrupted. Shutting down.")
                break

def main(args=None):
    rclpy.init(args=args)
    node = TurtleBotAgentNode()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
