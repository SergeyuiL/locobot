#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import apriltag
import numpy as np

class ImageSaverAndTagDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/locobot/camera/color/image_raw", Image, self.callback)
        self.image_saved = False

    def callback(self, data):
        if not self.image_saved:
            try:
                # Convert ROS Image message to OpenCV image
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                
                # Save the image
                cv2.imwrite("saved_image.jpg", cv_image)
                rospy.loginfo("Image saved to saved_image.jpg")
                
                self.detect_apriltag_family(cv_image)
                
                self.image_saved = True
            except CvBridgeError as e:
                rospy.logerr("CvBridge Error: {0}".format(e))

    def detect_apriltag_family(self, image):
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # List of possible AprilTag families
        families = ['tag36h11', 'tag25h9', 'tag16h5', 'tagCircle21h7', 'tagCircle49h12', 'tagStandard41h12', 'tagCustom48h12']
        
        detected_family = None

        # Iterate over each family to detect AprilTag
        for family in families:
            options = apriltag.DetectorOptions(families=family)
            detector = apriltag.Detector(options)
            results = detector.detect(gray)

            if results:
                detected_family = family
                for result in results:
                    rospy.loginfo(f"Detected AprilTag with ID: {result.tag_id} in family: {family}")
                break

        if detected_family is None:
            rospy.loginfo("No AprilTag detected.")
        else:
            rospy.loginfo(f"Detected AprilTag belongs to family: {detected_family}")

def main():
    rospy.init_node('image_saver_and_tag_detector', anonymous=True)
    ImageSaverAndTagDetector()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")

if __name__ == '__main__':
    main()
