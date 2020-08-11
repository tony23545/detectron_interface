import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import from common libraries
import numpy as np 
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import rclpy
from rclpy.node import Node

from detectron_interface.srv import DetecSrv
from sensor_msgs.msg import Image, PointCloud2 
import numpy as np
import matplotlib.pyplot as plt


class detectron_srv(Node):
    def __init__(self):
        super().__init__('detection_srv')
        # setup detectron model
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)
        #self.srv = self.create_service(DetecSrv, 'detection_srv', self.runPredictor)

        # subscribe to sensor 
        self.subscription = self.create_subscription(
            PointCloud2,
            '/realsense/camera/pointcloud',
            self.callback,
            30)
        self.count = -1

    def outlier_filter(self, x, z, idx):
        x_mean = np.mean(x)
        x_var = np.var(x)
        z_mean = np.mean(z)
        z_var = np.var(z)
        gaussian_kernel = np.exp(-0.5 * (np.power(x-x_mean, 2) / x_var + np.power(z-z_mean, 2) / z_var)) / (2 * np.pi * np.sqrt(x_var * z_var))
        return idx[gaussian_kernel > 0.1]


    def callback(self, msg):
        # self.count += 1
        # if self.count % 2 > 0:
        #     return
        print("processing one frame...")

        height = msg.height
        width = msg.width
        points = np.array(msg.data, dtype = 'uint8')
        # rgb image
        rgb_offset = msg.fields[3].offset
        point_step = msg.point_step
        r = points[rgb_offset::point_step]
        g = points[(rgb_offset+1)::point_step]
        b = points[(rgb_offset+2)::point_step]
        img = np.concatenate([r[:, None], g[:, None], b[:, None]], axis = -1)
        img = img.reshape((height, width, 3))
        # point cloud
        points = points.view('<f4')
        down_sample_scale = 16
        x = points[::int(down_sample_scale  * point_step / 4)]
        y = points[1::int(down_sample_scale * point_step / 4)]
        z = points[2::int(down_sample_scale * point_step / 4)]

        # call detectron model
        outputs = self.predictor(img)

        # map to point cloud data
        color = np.zeros_like(x, dtype = 'uint8')
        num_classes = outputs['instances'].pred_classes.shape[0]
        masks = outputs["instances"].pred_masks.cpu().numpy().astype('uint8').reshape((num_classes, -1))[:, ::down_sample_scale]
        head_count = 0
        for i in range(num_classes):
            if outputs["instances"].pred_classes[i] == 0:
                idx = np.where(masks[i])[0]
                idx = self.outlier_filter(x[idx], z[idx], idx)
                head_count += 1
                color[idx] += head_count

        plt.clf()
        plt.subplot(1, 2, 1)
        plt.scatter(x, z, c = color, s = 0.1)
        plt.subplot(1, 2, 2)
        plt.imshow(np.flip(img, 2))
        plt.draw()
        plt.pause(0.01)

    def runPredictor(self, request, response):
        height = request.rgb_img.height
        width = request.rgb_img.width
        img = np.array(request.rgb_img.data).reshape((height, width, 3))
        img = np.flip(img, 2)
        outputs = self.predictor(img)
        
        '''
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("outputs", out.get_image()[:, :, ::-1])
        #cv2.imshow("outputs", img)
        cv2.waitKey(1)'''
        return response

def main():
    rclpy.init(args = None)
    subs = detectron_srv()
    print("start spining detectron_srv node...")
    rclpy.spin(subs)

    subs.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()