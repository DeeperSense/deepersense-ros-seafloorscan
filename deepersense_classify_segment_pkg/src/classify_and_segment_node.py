#! /usr/bin/env python3.8

from predict_patches import PredictPatch

from cv_bridge import CvBridge
import rospy 

import threading 
from datetime import datetime 
import numpy as np
import matplotlib.pyplot as plt 
from sensor_msgs.msg import Image
from deepersense_msgs.msg import Patch
from deepersense_msgs.msg import Prediction
from nav_msgs.msg import GridCells
from geometry_msgs.msg import Point 
import cv2

import os 


class ClassifyAndSegment:

    def __init__(self):

        self.bridge = CvBridge()

        self.init_pos = None 

        # load params from ROS param server 
        patch_shape_x = rospy.get_param("/sonar/patch_shape_x")
        patch_shape_y = rospy.get_param("/sonar/patch_shape_y")
        stride_x = rospy.get_param("/sonar/stride_x")
        stride_y = rospy.get_param("/sonar/stride_y")
        max_pings = rospy.get_param("/undistortion/max_pings")
        num_samples = rospy.get_param("/sonar/num_samples")

        node_name = rospy.get_name()
        model_path = rospy.get_param(node_name + "/model_path")
        config_path = rospy.get_param(node_name + "/config_path")

        encoder = rospy.get_param("/prediction/encoder")
        decoder = rospy.get_param("/prediction/decoder")
        
        self.resolution = rospy.get_param("/visualisation/resolution")
        self.publish_output_image = rospy.get_param("/visualisation/publish_prediction")
        self.publish_output_grid_cells = rospy.get_param("/visualisation/publish_grid_cells")
        
        self.correct_waterfall = rospy.get_param("/undistortion/correct_waterfall")
        self.publish_undistortion = rospy.get_param("/visualisation/publish_undistortion");
        self.publish_waterfall = rospy.get_param("/visualisation/publish_waterfall");

        self.predict_patches = PredictPatch((patch_shape_y, patch_shape_x), (stride_y, stride_x), \
                                            (max_pings, num_samples), model_path, \
                                            config_path, encoder, decoder)

        self.image_count = 0
        
        self.patches_combined = None
        self.confidences_combined = None
        self.current_height = 0
        
        self.interpolation_window = rospy.get_param("/undistortion/interpolation_window")
        self.max_pings = rospy.get_param("/undistortion/max_pings")
        self.num_samples = None 
        self.ground_bins = None 
        self.blind_bins = None 
        
        # create output file name for the current XTF's combined rgb outputs 
        output_file_name = rospy.get_param("/simulator/nav_rosbag_file").split(".")[-2] + ".png"
        output_dir = rospy.get_param(rospy.get_name() + "/output_dir")
        self.output_file_path = output_dir + output_file_name 

        # initialise publishers and subscribers 
        self.sub = rospy.Subscriber("/patches", Patch, self.waterfall_cb)
        self.image_pub = rospy.Publisher("/prediction/image", Image, queue_size=20)
        self.output_pub = rospy.Publisher("/prediction/output", Prediction, queue_size=20)
        self.original_waterfall_pub = rospy.Publisher("/waterfall/distorted", Image, queue_size=1)
        self.corrected_waterfall_pub = rospy.Publisher("/waterfall/undistorted", Image, queue_size=1)

        self.class_grid_map = {}
        self.grids_per_class = {}

        # initialise grid cell messages for visualisation 
        for class_idx in list(self.predict_patches.cmap.keys()):
            self.class_grid_map[class_idx] = rospy.Publisher("/grid_cells/class_" + str(class_idx), GridCells, queue_size=100)

            self.grids_per_class[class_idx] = GridCells()
            self.grids_per_class[class_idx].header.frame_id = "world"
            self.grids_per_class[class_idx].cell_height = self.resolution
            self.grids_per_class[class_idx].cell_width = self.resolution


    def update_final_result(self, confidences, patches):
        """Updating rgb output image made up of combined waterfalls from the XTF 

        Args:
            confidences (_type_): confidence images 
            patches (_type_): patch images  
        """
        
        start = datetime.now()
        if (self.current_height == 0):
            self.current_height = self.max_pings
            self.confidences_combined = confidences
            self.patches_combined = patches 
            return

        all_confidences = self.confidences_combined
        all_patches = self.patches_combined 
        curr_confidences = confidences
        curr_patches = patches 

        all_patches_overlap = slice(self.current_height + self.interpolation_window - self.max_pings, self.current_height)
        curr_patch_overlap = slice(0, self.max_pings - self.interpolation_window)

        A = np.concatenate( (np.expand_dims(all_confidences[all_patches_overlap,:], axis=0), 
                             np.expand_dims(curr_confidences[curr_patch_overlap,:], axis=0) ), axis = 0)
        B = np.argmax(A, axis=0)
        y0, x0 = np.where(B==0)
        y1, x1 = np.where(B==1)

        new_height = self.current_height + self.interpolation_window
        C = np.zeros((new_height , self.num_samples))
        D = np.zeros((new_height, self.num_samples, 3))

        C[: self.current_height + self.interpolation_window - self.max_pings, :] = all_confidences[: self.current_height + self.interpolation_window - self.max_pings, :]
        C[new_height  - self.interpolation_window:, :] = curr_confidences[self.max_pings - self.interpolation_window:, :]
        D[: self.current_height + self.interpolation_window - self.max_pings, :, :] = all_patches[: self.current_height + self.interpolation_window - self.max_pings, :, :]
        D[new_height - self.interpolation_window:, :, :] = curr_patches[self.max_pings - self.interpolation_window:, :, :]

        C[self.current_height + self.interpolation_window  - self.max_pings: new_height  - self.interpolation_window][y0, x0] = all_confidences[all_patches_overlap][y0, x0]
        D[self.current_height + self.interpolation_window  - self.max_pings: new_height  - self.interpolation_window, :][y0, x0] = all_patches[all_patches_overlap, :][y0, x0, :]

        C[self.current_height + self.interpolation_window - self.max_pings: new_height  - self.interpolation_window][y1, x1] = curr_confidences[curr_patch_overlap][y1, x1]
        D[self.current_height + self.interpolation_window - self.max_pings: new_height  - self.interpolation_window, :][y1, x1] = curr_patches[curr_patch_overlap, :][y1, x1, :]

        self.confidences_combined = C
        self.patches_combined = D
        
        self.current_height = new_height
        
        plt.imsave(self.output_file_path, self.patches_combined.astype(np.uint8))


    def predict(self, i, image):
        """Predicting image 

        Args:
            i (_type_): _description_
            image (_type_): _description_
            positions (_type_): _description_

        Returns:
            _type_: _description_
        """

        print(f"Predicting image #{i}")
        start = datetime.now()
        class_outputs, rgb, confidences = self.predict_patches.predict(image)
        time_diff = (datetime.now() - start).total_seconds()
        print(f"Time taken (in seconds) to make prediction #{i}: {time_diff}")     
        return class_outputs, rgb, confidences
        
    def publish_output(self, i, waterfalls, class_outputs, rgb, confidences, image, positions):
        """Publish distorted and undistorted waterfalls, model output with HSV encoding, and grid-cells 

        Args:
            i (_type_): waterfall index 
            waterfalls (_type_): list of waterfalls 
            class_outputs (_type_): class index output 
            rgb (_type_): rgb output 
            confidences (_type_): confidence output 
            image (_type_): original image 
            positions (_type_): positions of image intensities 
        """
        
        # publish outputs

        confidences_1d = confidences.reshape((-1,1))
        classes_1d = class_outputs.reshape((-1,1))
        height, width = class_outputs.shape[:2]

        prediction = Prediction()
        prediction.header.seq = i
        prediction.header.stamp = rospy.Time.now()
        prediction.confidences = np.squeeze(confidences_1d).tolist()
        prediction.outputs = np.squeeze(classes_1d.astype(np.uint8)).tolist()

        prediction.height = height 
        prediction.width = width 
        self.output_pub.publish(prediction)

        # visualisation 

        if self.publish_output_image:
            gray = waterfalls[1] if self.correct_waterfall else waterfalls[0]
            encoded_rgb, encoded_hsv = PredictPatch.prediction_to_hsv_encoding(gray, rgb, confidences)
            msg = self.bridge.cv2_to_imgmsg(encoded_rgb, "rgb8")
            self.image_pub.publish(msg)

        if waterfalls[1] is not None and self.publish_undistortion:
            self.corrected_waterfall_pub.publish( self.bridge.cv2_to_imgmsg(waterfalls[1], encoding="mono8") )
            
        if waterfalls[0] is not None and self.publish_waterfall:
            self.original_waterfall_pub.publish( self.bridge.cv2_to_imgmsg(waterfalls[0], encoding="mono8") )
            
        if self.publish_output_grid_cells:
            
            start = datetime.now()

            # calculate positions for each 
            xs = (np.floor(positions[:,0] / self.resolution)).astype(int)
            ys = (np.floor(positions[:,1] / self.resolution)).astype(int)

            combined = np.column_stack((xs, ys))
            _, unique_idxs = np.unique(combined, axis=0, return_index=True)

            for idx in unique_idxs:
                point = Point()
                point.x = combined[idx, 0] * self.resolution 
                point.y = combined[idx, 1] * self.resolution 
                point.z = 0.0  

                self.grids_per_class[classes_1d[idx][0]].cells.append(point)

            for class_idx in self.class_grid_map:
                self.class_grid_map[class_idx].publish(self.grids_per_class[class_idx])

            time_diff = (datetime.now() - start).total_seconds()
            
        self.update_final_result(confidences, rgb)
        
    def process_waterfall_data(self, data, height, width, distorted):
        """Take logarithm of pixel intensities, perform min-max scaling and scale to 0, 255 range

        Args:
            data (_type_): intensities 
            height (_type_): image height 
            width (_type_): image width 

        Returns:
            _type_: output image 
        """

        data2 = np.asarray(data)
        image2 = data2.reshape((height, width))
        image2 = ((image2 - np.min(image2)) / (np.max(image2) - np.min(image2)) * 255.0).astype(np.uint8)

        if distorted:
            folder_dir = "/home/jetson/catkin_ws/src/deepersense_classify_segment_pkg/output/waterfalls/distorted/"
            image_name = "waterfall_" + str(len(os.listdir(folder_dir))) + ".png"
            image_path = os.path.join(folder_dir, image_name)
            cv2.imwrite(image_path, image2)

        else:
            folder_dir = "/home/jetson/catkin_ws/src/deepersense_classify_segment_pkg/output/waterfalls/undistorted/"
            image_name = "waterfall_" + str(len(os.listdir(folder_dir))) + ".png"
            image_path = os.path.join(folder_dir, image_name)
            cv2.imwrite(image_path, image2)

        data = np.asarray(data) 
        data = np.log10(data + 1e-4)
        data.clip(0, data.max(), out=data)
        image = data.reshape((height, width))
        image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255.0).astype(np.uint8)
        return image 
    
    def waterfall_cb(self, msg):
        """Receive data from topic, run model inference and publish outputs to other ROS topics 

        Args:
            msg (_type_): input ROS message 
        """
        
        # grab distorted and undistorted waterfalls and store them in global variables 
        waterfalls = [None]*2
        
        if self.correct_waterfall:
            waterfalls[1] = self.process_waterfall_data(msg.undistorted, msg.height, msg.width, False)
        waterfalls[0] = self.process_waterfall_data(msg.distorted, msg.height, msg.width, True)               
       
        if self.init_pos is None:
            self.init_pos = [msg.positions[0].x, msg.positions[0].y, msg.positions[0].z]
       
        # reshape intensity data into true image shape
        data = np.asarray(msg.undistorted) if self.correct_waterfall else np.asarray(msg.distorted)
        image = data.reshape((msg.height, msg.width))

        # apply min max scaling and convert to 0, 255 range 
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255.0
        image = image.astype(np.uint8)

        # reshape intensity positions in true image shape 
        positions = np.array([[point.x, point.y, point.z] for point in msg.positions])
        positions_reshaped = positions.reshape(msg.height, msg.width, 3)

        num_blind_bins = msg.blind_bins
        num_ground_bins = msg.ground_bins 

        # calculate water column indices (middle of waterfall)
        if self.num_samples == None:
            self.blind_bins = num_blind_bins 
            self.ground_bins = int((image.shape[1] - (self.blind_bins * 2)) / 2.0)
            self.num_samples = self.ground_bins * 2

        # extract port and starboard sections without water column
        port_idx = slice(0, self.ground_bins)
        stbd_idx = slice(self.ground_bins + 2 * self.blind_bins, image.shape[1])

        # create intensity and positions images without the watercolumn values  
        image_sliced = np.hstack((image[:, port_idx], image[:, stbd_idx]))
        positions_sliced = np.hstack((positions_reshaped[:, port_idx, :], positions_reshaped[:, stbd_idx, :])).reshape((-1,3))
        
        # apply same modification to the distorted and undistorted images 
        for i in range(2):
            if waterfalls[i] is not None:
                waterfalls[i] = np.hstack((waterfalls[i][:, port_idx], waterfalls[i][:, stbd_idx]))
        
        # perform prediction and then 
        current_index = self.image_count
        class_outputs, rgb, confidences = self.predict(current_index, image_sliced)
        x = threading.Thread(target=self.publish_output, args=(current_index, waterfalls, class_outputs, rgb, confidences, image_sliced, positions_sliced), daemon=True)
        x.start()

        self.image_count += 1


if __name__ == '__main__':
    rospy.init_node("classify_segment_node")
    cs = ClassifyAndSegment()

    rospy.spin()
