#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from langchain_ollama import ChatOllama
from langchain.agents import tool
from rosa import ROSA, RobotSystemPrompts
import math
import time
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import threading


class TurtleBotAgentNode(Node):
    def __init__(self):
        super().__init__('turtlebot_agent_node')
        
        # ============================= INITIALIZE NODE =============================
        # Initialize publishers
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Initialize CV Bridge for image processing
        self.bridge = CvBridge()
        
        # ============================= SETUP AGENT =============================
        # Setup the agent
        self.setup_agent()
        
        self.get_logger().info("ROSA TurtleBot Agent is ready. Type a command:")

    def setup_agent(self):
        """Setup the ROSA agent with LLM and tools"""
        # ============================= INITIALIZE LLM =============================
        # Initialize the local LLM (Ollama)
        local_llm = self._initialize_llm()
        
        # ============================= DEFINE TOOLS & PROMPTS =============================
        # Define tools
        tools = self._create_tools()
        
        # Setup agent prompts
        prompts = self._create_prompts()
        
        # ============================= INITIALIZE ROSA AGENT =============================
        # Initialize ROSA
        self.agent = ROSA(
            ros_version=2,
            llm=local_llm,
            tools=tools,
            prompts=prompts
        )

    def _initialize_llm(self):
        """Initialize and configure the local LLM"""
        return ChatOllama(
            model="llama3.1:8b",
            temperature=0.0,
            max_retries=2,
            num_ctx=8192,
        )
    
    def _create_prompts(self):
        """Create the system prompts for the agent"""
        return RobotSystemPrompts(
            embodiment_and_persona="You are a smart TurtleBot in turtlesim with a camera.",
            about_your_capabilities=(
                "You have access to tools, and you should always prefer using them over replying directly. "
                "You can move forward and rotate left/right using degrees. "
                "You have a camera and can see the environment. "
                "Always use your available tools to answer questions and execute tasks. "
                "Never guess or use ROS commands directly."
            ),
            mission_and_objectives=(
                "Help users control the turtle and inspect the environment using available tools. "
                "When the user gives any command that involves 'camera', 'image', 'see', 'show me', or 'feed', "
                "you **must** call the `get_robot_camera_image` tool. Do not guess or provide manual instructions."
            )
        )
    
    def _create_tools(self):
        """Create and return all the tools for the agent"""
        node_instance = self  # Store reference to self for closure
        
        # ============================= MOVEMENT TOOLS =============================
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
            
                
            start_time = node_instance.get_clock().now().nanoseconds
            while rclpy.ok():
                current_time = node_instance.get_clock().now().nanoseconds
                elapsed = (current_time - start_time) / 1e9  # Convert ns to seconds
                if elapsed >= duration:
                    break
                node_instance.publisher_.publish(twist)
                rclpy.spin_once(node_instance, timeout_sec=0.01)

            twist.linear.x = 0.0
            node_instance.publisher_.publish(twist)
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

            start_time = node_instance.get_clock().now().nanoseconds
            while rclpy.ok():
                current_time = node_instance.get_clock().now().nanoseconds
                elapsed = (current_time - start_time) / 1e9
                if elapsed >= duration:
                    break
                node_instance.publisher_.publish(twist)
                rclpy.spin_once(node_instance, timeout_sec=0.01)

            twist.angular.z = 0.0
            node_instance.publisher_.publish(twist)
            return f"Rotated {angle:.0f}° to the {direction}."

        # ============================= SENSOR TOOLS =============================
        @tool
        def get_turtle_pose() -> dict:
            """
            Get the pose of the turtle (/turtle1).
            """
            pose_data = {}

            def callback(msg):
                pose_data["x"] = round(msg.x, 2)
                pose_data["y"] = round(msg.y, 2)
                pose_data["theta"] = round(msg.theta, 2)
                pose_data["linear_velocity"] = round(msg.linear_velocity, 2)
                pose_data["angular_velocity"] = round(msg.angular_velocity, 2)

            sub = node_instance.create_subscription(
                Pose,
                "/turtle1/pose",
                callback,
                10
            )
            # Wait for the message with a timeout (max 5 seconds)
            timeout = 5
            start_time = time.time()
            while time.time() - start_time < timeout:
                if pose_data:
                    break
                rclpy.spin_once(node_instance, timeout_sec=0.1)

            if not pose_data:
                return {"error": "Pose not received in time."}

            return pose_data

        # ============================= CAMERA TOOLS =============================
        @tool
        def get_robot_camera_image() -> dict:
            """
            Display a live stream from the TurtleBot's RGB camera (/turtlebot_rgb).
            Press 'q' to close the stream window.
            """
            streaming = {"running": True}
            last_frame = {"image": None}

            def image_callback(msg):
                try:
                    cv_image = node_instance.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    resized_image = cv2.resize(cv_image, (250, 250))
                    last_frame["image"] = resized_image
                except Exception as e:
                    node_instance.get_logger().error(f"Image conversion failed: {str(e)}")

            sub = node_instance.create_subscription(
                Image,
                "/turtlebot_rgb",
                image_callback,
                10
            )

            node_instance.get_logger().info("📷 Live streaming camera... Press 'q' to quit.")
            result = {}

            try:
                while rclpy.ok() and streaming["running"]:
                    rclpy.spin_once(node_instance, timeout_sec=0.01)

                    frame = last_frame["image"]
                    if frame is not None:
                        cv2.imshow("Live TurtleBot Camera View", frame)
                        key = cv2.waitKey(1)
                        if key == ord('q'):
                            streaming["running"] = False
            except Exception as e:
                result["error"] = f"Error during streaming: {str(e)}"
            finally:
                node_instance.destroy_subscription(sub)
                cv2.destroyAllWindows()
                if "error" not in result:
                    result["message"] = "Stopped live stream."

            return result
            
        # Return all tools
        return [
            publish_linear_motion,
            publish_angular_motion,
            get_turtle_pose,
            get_robot_camera_image
        ]


# ============================= MAIN FUNCTION =============================
def main(args=None):
    rclpy.init(args=args)
    node = TurtleBotAgentNode()

    try:
        # ============================= INTERACTIVE COMMAND LOOP =============================
        while rclpy.ok():
            user_input = input("🧠 Your command > ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            response = node.agent.invoke(user_input)
            print(f"🤖 ROSA: {response}")
    except KeyboardInterrupt:
        print("\n[!] Interrupted. Shutting down.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()