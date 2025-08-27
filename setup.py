from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'deblur_gan_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'weights'), glob('ghost_deblurgan/trained_weights/*')),
        (os.path.join('share', package_name, 'config'), glob('ghost_deblurgan/config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Emanuele Nencioni',
    maintainer_email='emanuele.nencioni1@gmail.com',
    description='ROS2 node for Ghost-DeblurGAN',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'deblur_gan_node = deblur_gan_ros.deblur_gan_ros:main'
        ],
    },
)