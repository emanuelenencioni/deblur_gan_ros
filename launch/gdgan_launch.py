from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='deblur_gan_ros',
            executable='deblur_gan_node',
            name='deblur_gan_ros',
            output='screen',
            parameters=[
                {'input_topic': 'image_raw'},
                {'output_topic': 'deblur_gan/image_raw'},
                {'weights_name': 'fpn_ghostnet_gm_hin.h5'},
                {'model_name': 'fpn_ghostnet'},
                {'cuda': True}
            ]
        )
    ])
