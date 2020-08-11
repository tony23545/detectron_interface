import rclpy
from rclpy.node import Node

from detectron_interface.srv import DetecSrv
from sensor_msgs.msg import Image, PointCloud2

import cv2
import numpy as np
class testNode(Node):
    def __init__(self):
        super().__init__('testNode')
        self.client = self.create_client(DetecSrv, 'detection_srv')
        #while not self.client.wait_for_service(timeout_sec = 1.0):
        #    self.get_logger().info('service not available, waiting again...')

        self.subscription = self.create_subscription(
            PointCloud2,
            '/realsense/camera/pointcloud',
            self.pcd_callback,
            30)
        self.latest_img = None
        self.img_sub = self.create_subscription(
            Image,
            '/realsense/camera/color/image_raw', 
            self.img_callback,
            30)
        self.count = -1

    def img_callback(self, msg):
        img = np.array(msg.data, dtype = 'uint8')
        img = img.reshape((msg.height, msg.width, 3))
        self.latest_img = img

    def pcd_callback(self, msg):
        print("send request")
        self.count += 1
        if self.count % 30 > 0:
           return
        
        points = np.array(msg.data, dtype = 'uint8')
        # rgb image
        rgb_offset = msg.fields[3].offset
        point_step = msg.point_step
        r = points[rgb_offset::point_step]
        g = points[(rgb_offset+1)::point_step]
        b = points[(rgb_offset+2)::point_step]
        img = np.concatenate([r[:, None], g[:, None], b[:, None]], axis = -1)
        img = img.reshape((msg.height, msg.width, 3))
        #img = img.reshape((480, 640, 2))
        #img = np.flip(img, 2)
        cv2.imshow("img", img)
        cv2.waitKey(1)
        '''
        req = DetecSrv.Request()
        req.rgb_img = msg
        self.future = self.client.call_async(req)
        print("get response")'''

def main():
    rclpy.init(args = None)
    subs = testNode()
    print("start spining test node...")
    rclpy.spin(subs)

    subs.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()