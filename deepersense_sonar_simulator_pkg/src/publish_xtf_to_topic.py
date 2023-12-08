#! /usr/bin/env python3.8

import rospy 
import pyxtf
from deepersense_msgs.msg import Ping, PingInfo
from cola2_msgs.msg import NavSts
import numpy as np 
import os 
from datetime import datetime 
import time 
import subprocess
import rosbag
from pyproj import CRS, Proj 

class SonarSimulator:

    def __init__(self):
        dir_name = rospy.get_param(rospy.get_name() + "/config_dir_path")
        xtf_file_name = rospy.get_param("/simulator/xtf_file")
        ping_start_idx = rospy.get_param("/simulator/ping_start")
        
        merged_folder = "merged"
        nav_folder = "navigation"
        xtf_folder = "sonar"

        self.pings_topic = rospy.get_param("/simulator/pings_topic_name")
        self.ping_info_topic = rospy.get_param("/simulator/ping_info_topic_name")
        
        xtf_file_path = os.path.join(dir_name, xtf_folder, rospy.get_param("/simulator/xtf_file"))
        self.sonar_pings, self.info, self.sonar_image = self.load_xtf(xtf_file_path, ping_start_idx)

        publish_rate = rospy.get_param("/simulator/publish_rate")
        combined = rospy.get_param("/simulator/separate_inputs")

        if combined:
            nav_file_name = rospy.get_param("/simulator/nav_rosbag_file")
            nav_file_path = os.path.join(dir_name, nav_folder, nav_file_name)
            print(f"Navigation file path: {nav_file_path}")
            
            merged_rosbag_name = nav_file_name.split(".")[-2] + "_with_sonar_data.bag" 
            output_bag_file_path = os.path.join(dir_name, merged_folder, merged_rosbag_name)     
            print(f"Merged XTF and navigation file path: {output_bag_file_path}")
            
            nav_topic = rospy.get_param("/simulator/nav_topic_name")
            self.publish_nav_xtf(nav_file_path, nav_topic, output_bag_file_path)
        else:
            self.publish_xtf(publish_rate)

    def load_xtf(self, xtf_file_path, start_idx=0):
        """Extracts data from XTF file 

        Args:
            xtf_file_path (_type_): file path 
            start_idx (int, optional): xtf starting ping index. Defaults to 0.

        Returns:
            _type_: pings, ping info and sonar image 
        """

        xtf_file_name = xtf_file_path.split("/")[-1]
        rospy.loginfo(f"XTF file name: {xtf_file_name}")

        # load xtf file 
        (_, packet) = pyxtf.xtf_read(xtf_file_path)
        sonar_pings = packet[pyxtf.XTFHeaderType.sonar][start_idx:]    
        print(f"Number of pings: {len(sonar_pings)}")

        sonar_chans = [np.vstack([ping.data[i] for ping in sonar_pings]) for i in range(2)]
        sonar_image = np.hstack((sonar_chans[0], sonar_chans[1]))
        
        ping_info = sonar_pings[0].ping_chan_headers[0]
        num_samples = ping_info.NumSamples*2
        slant_range = ping_info.SlantRange         
        slant_res = slant_range*2/num_samples

        rospy.loginfo(f"Number of samples: {num_samples}")
        rospy.loginfo(f"Slant range: {slant_range}")
        rospy.loginfo(f"Slant resolution: {slant_res}")

        info = PingInfo()
        info.slant_range = slant_range
        info.num_samples = num_samples
        info.slant_res = slant_res

        return sonar_pings, info, sonar_image

    def publish_xtf(self, publish_rate):
        """Publish data inside xtf to ROS topics

        Args:
            publish_rate (_type_): Rate (Hz) at which to publish 
        """

        rate = rospy.Rate(publish_rate)
        
        lonlat2EN = Proj(CRS.from_epsg(25831), preserve_units=False)

        ping_pub = rospy.Publisher(self.pings_topic, Ping, queue_size=200)
        info_pub = rospy.Publisher(self.ping_info_topic, PingInfo, queue_size=1)
    
        i = 0

        trajectory = [(ping.SensorXcoordinate, ping.SensorYcoordinate, ping.SensorPrimaryAltitude,
                        ping.SensorRoll, ping.SensorPitch, ping.SensorHeading) for ping in self.sonar_pings]
        E, N, h, r, p, y = zip(*trajectory) 
        
        r = np.radians(r)
        p = np.radians(p)
        y = np.radians(y)
        

        while not rospy.is_shutdown():
            if i == len(E):
                break 

            ping = Ping()

            ping.header.stamp = rospy.Time.now()
            ping.header.seq = i
            ping.header.frame_id = "world"

            ping.altitude = h[i]
            ping.pitch = p[i]
            ping.roll = r[i]
            ping.yaw = y[i]
            ping.x = E[i]
            ping.y = N[i]

            ping.intensities = self.sonar_image[i,:].tolist()

            ping_pub.publish(ping)
            info_pub.publish(self.info)

            i += 1
            rate.sleep()
        
        rospy.logwarn("XTF finished! No more data to publish.")

    def find_common_times(self, bag):
        """Find intersection time interval 

        Args:
            bag (_type_): ROS bag object 

        Returns:
            _type_: start and final time of interval
        """

        start_time_sonar = datetime(self.sonar_pings[0].Year, self.sonar_pings[0].Month, 
                                self.sonar_pings[0].Day, self.sonar_pings[0].Hour, 
                                self.sonar_pings[0].Minute, self.sonar_pings[0].Second, 
                                int(self.sonar_pings[0].HSeconds * 1e4))
            
        end_time_sonar = datetime(self.sonar_pings[len(self.sonar_image)-1].Year, self.sonar_pings[len(self.sonar_image)-1].Month, 
                                self.sonar_pings[len(self.sonar_image)-1].Day, self.sonar_pings[len(self.sonar_image)-1].Hour, 
                                self.sonar_pings[len(self.sonar_image)-1].Minute, self.sonar_pings[len(self.sonar_image)-1].Second, 
                                int(self.sonar_pings[len(self.sonar_image)-1].HSeconds * 1e4))

        start_time_nav = datetime.utcfromtimestamp(bag.get_start_time())
        end_time_nav  = datetime.utcfromtimestamp(bag.get_end_time())
        
        start_time = start_time_sonar if start_time_sonar > start_time_nav else start_time_nav
        end_time = end_time_sonar if end_time_sonar < end_time_nav else end_time_nav

        return start_time, end_time
    
    def ping_to_datetime(self, ping):
        """Get datetime object from ping date and time information

        Args:
            ping (_type_): ping 

        Returns:
            _type_: datetime object 
        """

        return datetime(ping.Year, ping.Month, ping.Day, ping.Hour, 
                                    ping.Minute, ping.Second, int(ping.HSeconds * 1e4))

    def publish_nav_xtf(self, input_bag_file_path, nav_topic, output_bag_file_path):
        """Publishes the navigation data from robag in one topic and xtf data in another topics

        Args:
            input_bag_file_path (_type_): bag file path
            nav_topic (_type_): navigation topic inside rosbag 
            output_bag_file_path (_type_): output bag file path 
        """

        lonlat2EN = Proj(CRS.from_epsg(25831), preserve_units=False)
        
        # if merged rosbag does not exist 
        exists = os.path.exists(output_bag_file_path)
        if not exists:

            input_bag = rosbag.Bag(input_bag_file_path)
            start_time, end_time = self.find_common_times(input_bag)

            # extract pings within time interval
            trajectory = [(ping.SensorPrimaryAltitude, self.ping_to_datetime(ping).timestamp(), i) 
                                    for i, ping in enumerate(self.sonar_pings)
                                    if ((self.ping_to_datetime(ping) >= start_time) & (self.ping_to_datetime(ping) <= end_time))]
            
            h, sonar_times, ping_idxs = zip(*trajectory) 

            # extract navigation within time interval
            nav_data = []
            nav_times = []
            for _, msg, t in input_bag.read_messages(topics=[nav_topic]):
                if (datetime.utcfromtimestamp(t.to_sec()) > end_time):
                    break

                if (datetime.utcfromtimestamp(t.to_sec()) >= start_time):
                    nav_data.append(msg)       
                    nav_times.append(datetime.utcfromtimestamp(t.to_sec()).timestamp())
            
            input_bag.close()
            num_sonar_msgs = len(sonar_times)

            # combine times and sort by time
            combined_times  = list(sonar_times) + nav_times
            sorted_idxs = np.argsort(combined_times)
             
            bag = rosbag.Bag(output_bag_file_path, "w")
            for idx in sorted_idxs:
                if idx >= num_sonar_msgs:
                    real_idx = idx - num_sonar_msgs
                    nav = nav_data[real_idx]
                    nav.header.stamp = rospy.Time(nav_times[real_idx])
            
                    bag.write(nav_topic, nav, nav.header.stamp)
                else:
                    ping = Ping()

                    ping.header.stamp = rospy.Time.from_sec(sonar_times[idx])
                    ping.header.seq = idx
                    ping.header.frame_id = "world"

                    ping.altitude = h[idx]
                    ping.intensities = self.sonar_image[ping_idxs[idx],:].tolist()

                    bag.write(self.pings_topic, ping, ping.header.stamp)
                    bag.write(self.ping_info_topic, self.info, ping.header.stamp)
            bag.close()
        
        time.sleep(5)
        subprocess.Popen(['rosbag', 'play', '--quiet', '-s', '102.0', output_bag_file_path])


if __name__ == '__main__':
    rospy.init_node("simulator_node")
    sim = SonarSimulator()
    rospy.spin()
