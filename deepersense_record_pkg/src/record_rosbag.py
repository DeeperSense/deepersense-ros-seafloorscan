#! /usr/bin/env python3.8

import os 
import rospy 
import subprocess, shlex, psutil
import datetime 

def load_topics_to_record(file_path):
	"""Loads topics from text file to record 

	Args:
		file_path (_type_): text file path 

	Returns:
		_type_: list of topic names 
	"""
	if file_path.split(".")[-1] != "txt":
		rospy.logerr("Wrong file extension") 
	
	topics = []
	with open(file_path, 'r') as f:
  		lines = []
  		for line in f:
  			record_topic = int(line.split(":")[-1].rstrip())
  			if record_topic:
  				topics.append(line.split(":")[-2].rstrip())
	return topics		
	
def record_rosbag(rosbag_path, topics):
	"""Record rosbag with specific topics 

	Args:
		rosbag_path (_type_): rosbag file path 
		topics (_type_): list of names of topics
	"""
	command = f"rosbag record -O {rosbag_path}"
	for topic in topics:
		command += f" {topic}"
	
	command = shlex.split(command)
	rosbag_proc = subprocess.Popen(command)
	
if __name__ == '__main__':
	rospy.init_node("rosbag_record_node")
	
	now = datetime.datetime.now()
	rosbag_name = f"deepersense_survey_{now.year}_{now.month}_{now.day}-{now.hour}_{now.minute}_{now.second}.bag"
	
	topics_file_path = rospy.get_param(rospy.get_name() + "/topics_file_path")
	rosbag_out_file_path = os.path.join(rospy.get_param(rospy.get_name() + "/rosbag_output_dir"), rosbag_name)
	
	topics = load_topics_to_record(topics_file_path)
	record_rosbag(rosbag_out_file_path, topics)
	
	rospy.spin()
