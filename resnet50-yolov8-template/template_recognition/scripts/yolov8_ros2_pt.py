#!/usr/bin/env python3

from ultralytics import YOLO  
import rclpy  
from rclpy.node import Node  
from cv_bridge import CvBridge 
from sensor_msgs.msg import Image  
from yolov8_template_msgs.msg import Yolov8Inference, InferenceResult 
import os 
import cv2
import numpy as np
import tensorflow as tf


#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU usage for TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging noise
cv_bridge = CvBridge()

# Class names (must match training labels)
YOLOV8_CLASSES = [
    'camouflage_soldier', 'weapon', 'military_tank', 'military_truck',
    'military_vehicle', 'civilian', 'soldier', 'civilian_vehicle',
    'military_artillery', 'trench', 'military_aircraft', 'military_warship'
]
RESNET_CLASSES = [
    'aircraft', 'artillery', 'camouflage', 'civilian', 'civilian_vehicle',
    'military_vehicle', 'soldier', 'tank', 'truck', 'warship', 'weapon'
]
class camera_sub(Node):
    
    def __init__(self):
        super().__init__('camera_subscriber')

        # Load YOLOv8 model
        yolo_path = os.path.expanduser('~/ros2_ws/src/ros2-yolov8-template/template_recognition/scripts/yolov8n.pt')
        self.yolo_model = YOLO(yolo_path)
        self.classes_to_detect = list(range(12))

        # Load ResNet50 model
        resnet_path = os.path.expanduser('~/ros2_ws/src/ros2-yolov8-template/template_recognition/scripts/resnet50.keras')
        self.keras_model = tf.keras.models.load_model(resnet_path)

        # Message and publishers
        self.yolov8_inference = Yolov8Inference()
        self.subscription = self.create_subscription(
            Image,
            '/bcr_bot/kinect_camera/image_raw',
            self.camera_callback,
            10
        )
        self.yolov8_pub = self.create_publisher(Yolov8Inference, "/Yolov8_Inference", 1)
        self.img_pub = self.create_publisher(Image, "/inference_result", 1)

    def preprocess_roi(self, roi):
        resized = cv2.resize(roi, (224, 224))  #based on my model input size
        normalized = resized / 255.0  # Normalize if your model expects this
        return normalized.reshape(1, 224, 224, 3)  # Add batch dimension

    def camera_callback(self, data):
        img = cv_bridge.imgmsg_to_cv2(data, "bgr8")
        results = self.yolo_model(img, classes=self.classes_to_detect)

        self.yolov8_inference.header.frame_id = "inference"
        self.yolov8_inference.header.stamp = self.get_clock().now().to_msg()

        for r in results:
            boxes = r.boxes 
            for box in boxes:
                inference_result = InferenceResult()
                b = box.xyxy[0].to('cpu').detach().numpy().copy()  
                c = box.cls  

                x1, y1, x2, y2 = map(int, b)

                # Expand the ROI by adding padding
                padding = 50  # Adjust this value based on your needs
                x1 = max(0, x1 - padding)  # Ensure bounds are within image dimensions
                y1 = max(0, y1 - padding)
                x2 = min(img.shape[1], x2 + padding)
                y2 = min(img.shape[0], y2 + padding)


                roi = img[x1:x2, y1:y2]
                if roi.size == 0:
                    continue

                # Visualize ROI
                cv2.imshow("ROI Passed to ResNet", roi)
                cv2.waitKey(1)

                # Preprocess ROI for Keras model
                roi_input = self.preprocess_roi(roi)

                # Run ResNet classification
                prediction = self.keras_model.predict(roi_input, verbose=0)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)

                # Log results
                yolo_class = YOLOV8_CLASSES[int(c)]
                resnet_class = RESNET_CLASSES[predicted_class]
                self.get_logger().info(f"YOLO Class: {yolo_class}, ResNet Class: {resnet_class} (Confidence: {confidence:.2f})")

                if yolo_class != resnet_class:
                    self.get_logger().warn(f"Mismatch: YOLO ({yolo_class}) ≠ ResNet ({resnet_class})")


                # Set result fields
                inference_result.yolo_class = YOLOV8_CLASSES[int(c)]
                inference_result.resnet_class = RESNET_CLASSES[predicted_class]
                inference_result.top = x1
                inference_result.left = y1
                inference_result.bottom = x2
                inference_result.right = y2

                self.yolov8_inference.yolov8_inference.append(inference_result)

        annotated_frame = results[0].plot()
        img_msg = cv_bridge.cv2_to_imgmsg(annotated_frame)
        self.img_pub.publish(img_msg)
        self.yolov8_pub.publish(self.yolov8_inference)
        self.yolov8_inference.yolov8_inference.clear()

if __name__ == '__main__':
    rclpy.init(args=None)
    camera_subscriber = camera_sub()
    rclpy.spin(camera_subscriber)  # Keep the node alive to continue processing callbacks
    rclpy.shutdown()  # Shutdown ROS2 cleanly
