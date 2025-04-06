from launch import LaunchDescription
from launch.actions import ExecuteProcess
from pathlib import Path

def generate_launch_description():
    # Get the absolute path of the follow_gap.py script relative to this file
    script_path = Path(__file__).parent / 'reactive_node.py'

    return LaunchDescription([
        ExecuteProcess(
            cmd=['python3', str(script_path)],
            shell=True,
            output='screen'
        )
    ])
