from setuptools import setup
import os

package_name = 'turtlebot_agent'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        # Install marker file
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        # Install package.xml
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'jpl-rosa', 'langchain', 'langchain-ollama', 'python-dotenv'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='A ROS 2 package integrating ROSA with a local LLM to control TurtleBot3 in simulation.',
    license='Your License',
    entry_points={
        'console_scripts': [
            'turtlebot_agent_node = turtlebot_agent.turtlebot_agent_node:main',
        ],
    },
)
