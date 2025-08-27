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
                {'input_topic': '/zed/zed_node/right_raw/image_raw_color'},
                {'output_topic': 'deblur_gan/image_raw'},
                {'weights_name': 'fpn_ghostnet_gm_hin.h5'},
                {'model_name': 'fpn_ghostnet'},
                {'cuda': True}
            ]
        )
    ])
