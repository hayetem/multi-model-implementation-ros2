#!/usr/bin/env python3

import cv2 
import threading 
import rclpy  
from rclpy.node import Node 
from cv_bridge import CvBridge  
from sensor_msgs.msg import Image  
from yolov8_template_msgs.msg import Yolov8Inference 

cv_bridge = CvBridge()

class camera_sub(Node):

    def __init__(self):

        super().__init__('camera_subscriber')

        
        self.subscription = self.create_subscription(
            Image,
            '/bcr_bot/kinect_camera/image_raw',
            self.camera_callback,
            10) 


    def camera_callback(self, data):
        global img
        img = cv_bridge.imgmsg_to_cv2(data, "bgr8")

class yolo_sub(Node):

    def __init__(self):
        super().__init__('yolo_subscriber')

        self.subscription = self.create_subscription(
            Yolov8Inference,
            '/Yolov8_Inference',  
            self.yolo_callback,
            10) 

        self.cnt = 0 
        self.img_pub = self.create_publisher(Image, "/inference_result_cv2", 1) # Publisher for annotated images.

    def yolo_callback(self, data):
        global img
        for r in data.yolov8_inference:
            class_name = r.class_name
            top = r.top
            left = r.left
            bottom = r.bottom
            right = r.right
            yolo_subscriber.get_logger().info(f"{self.cnt} {class_name} : {top}, {left}, {bottom}, {right}")

            cv2.rectangle(img, (left, top), (right, bottom), (255, 255, 0), 2)

            self.cnt += 1

        self.cnt = 0 
        img_msg = cv_bridge.cv2_to_imgmsg(img) 
        self.img_pub.publish(img_msg)  

if __name__ == '__main__':
    rclpy.init(args=None)
    yolo_subscriber = yolo_sub()
    camera_subscriber = camera_sub()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(yolo_subscriber)
    executor.add_node(camera_subscriber)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    rate = yolo_subscriber.create_rate(2) 
    try:
        while rclpy.ok():
            rate.sleep()
    except KeyboardInterrupt:
      
        pass

    rclpy.shutdown() 
    executor_thread.join() 