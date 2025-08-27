import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
import cv2
import yaml
import os
import torch
import numpy as np
import sys



# Assuming Ghost-DeblurGAN modules are in the python path
from ghost_deblurgan.models.networks import get_generator
from ghost_deblurgan.aug import get_normalize

class Predictor:
    def __init__(self, weights_path: str, model_name: str = '', cuda: bool = True, logger=None):
        config_path = os.path.join(get_package_share_directory('deblur_gan_ros'), 'config', 'config.yaml')
        with open(config_path) as cfg:
            config = yaml.safe_load(cfg)
            if logger: logger.info("config loaded")
        if logger: logger.info(f'Loading model {model_name or config["model"]}...')

        model = get_generator(config["model"], cuda=cuda)
        model.load_state_dict(torch.load(weights_path)['model'])
        self.model = model.module.cpu() if not cuda else model.cuda()
        self.cuda = cuda
        self.model.train(True)
        self.normalize_fn = get_normalize()

    @staticmethod
    def _array_to_batch(x):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    def _preprocess(self, x: np.ndarray):
        x, _ = self.normalize_fn(x, x)
        h, w, _ = x.shape
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        pad_params = {'mode': 'constant',
                      'constant_values': 0,
                      'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
                      }
        x = np.pad(x, **pad_params)
        return self._array_to_batch(x), h, w

    @staticmethod
    def _postprocess(x: torch.Tensor) -> np.ndarray:
        x, = x
        x = x.detach().cpu().float().numpy()
        x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
        return x.astype('uint8')

    def __call__(self, img: np.ndarray) -> np.ndarray:
        (img_batch), h, w = self._preprocess(img)
        with torch.no_grad():
            inputs = [img_batch.cuda() if self.cuda else img_batch.cpu()]
            pred = self.model(*inputs)
        return self._postprocess(pred)[:h, :w, :]


class DeblurGanNode(Node):
    def __init__(self):
        super().__init__('deblur_gan_node')
        self.get_logger().info('python version: %s' % sys.version)
        self.declare_parameter('input_topic', 'image_raw')
        self.declare_parameter('output_topic', 'deblur_gan/image_raw')
        self.declare_parameter('weights_name', 'fpn_ghostnet_gm_hin.h5')
        self.declare_parameter('model_name', 'fpn_ghostnet')
        self.declare_parameter('cuda', True)

        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        weights_name = self.get_parameter('weights_name').get_parameter_value().string_value
        model_name = self.get_parameter('model_name').get_parameter_value().string_value
        self.cuda = self.get_parameter('cuda').get_parameter_value().bool_value

        self.subscription = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            10)
        self.publisher = self.create_publisher(Image, output_topic, 10)
        self.bridge = CvBridge()

        weights_path = os.path.join(get_package_share_directory('deblur_gan_ros'), 'weights', weights_name)

        self.predictor = Predictor(weights_path=weights_path, model_name=model_name, cuda=self.cuda, logger=self.get_logger())
        self.get_logger().info('deblur_gan_ros started')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error('Could not convert image: %s' % str(e))
            return

        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        deblurred_image = self.predictor(cv_image_rgb)
        deblurred_image_bgr = cv2.cvtColor(deblurred_image, cv2.COLOR_RGB2BGR)

        try:
            self.publisher.publish(self.bridge.cv2_to_imgmsg(deblurred_image_bgr, "bgr8"))
        except Exception as e:
            self.get_logger().error('Could not publish image: %s' % str(e))


def main(args=None):
    rclpy.init(args=args)
    deblur_gan_node = DeblurGanNode()
    rclpy.spin(deblur_gan_node)
    deblur_gan_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
