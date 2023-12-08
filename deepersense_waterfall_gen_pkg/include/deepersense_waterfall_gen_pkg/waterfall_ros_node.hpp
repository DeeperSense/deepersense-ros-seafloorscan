#pragma once 

#include <deepersense_msgs/Ping.h>
#include <deepersense_msgs/PingInfo.h>
#include <deepersense_msgs/Patch.h>

#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <Eigen/Core>
#include <cola2_msgs/NavSts.h>

#include <boost/bind.hpp>

#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Quaternion.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "waterfall_gen.hpp"

class WaterfallNode
{

public:
    WaterfallNode(ros::NodeHandle* nh, 
                  ros::NodeHandle* privateNh);
    ~WaterfallNode() {}

private:

    deepersense::SynthesizeWaterfall waterfallGen_;

    ros::NodeHandle nh_, privateNh_;

    ros::Subscriber pingSub_, pingInfoSub_;
    ros::Publisher patchPub_, rawSwathsPub_, correctedSwathsPub_;
    ros::Publisher originalWaterfallPub_, undistortedWaterfallPub_;
    ros::Publisher markerPub_;

    std::string pingsTopic_, pingInfoTopic_, navTopic_;

    typedef message_filters::sync_policies::ApproximateTime<deepersense_msgs::Ping, cola2_msgs::NavSts> SyncPolicy;
    typedef message_filters::Synchronizer<SyncPolicy> Sync;
    boost::shared_ptr<Sync> sync_;

    message_filters::Subscriber<deepersense_msgs::Ping> pingCombinedSub_;
    message_filters::Subscriber<cola2_msgs::NavSts> navCombinedSub_;

    bool visWaterfall_, visUndistortion_, visSwathLines_;

    int imageCount_;
    bool separateInputs_;
    bool correctWaterfall_;

    visualization_msgs::Marker robotMarker_;
    visualization_msgs::Marker rawSwaths_, correctedSwaths_;

    Eigen::Vector3f meshRotationOffset_;
    
    /**
     * @brief Processes incoming pings, deciding when to publish the accumulated data and when to correct the waterfall
     * 
     * @param x UTM x coordinate
     * @param y UTM y coordinate
     * @param altitude Altitude in metres
     * @param roll Roll in radians
     * @param pitch Pitch in radians 
     * @param yaw Yaw in radians 
     * @param intensities Intensity data vector 
     */
    void onPingCallback(double x, 
                    double y, 
                    double altitude, 
                    float roll, 
                    float pitch, 
                    float yaw, 
                    const std::vector<int>& intensities);

    /**
     * @brief Callback to the side-scan sonar ping information 
     * 
     * @param msg ROS PingInfo message 
     */
    void pingInfoCallback(const deepersense_msgs::PingInfo::ConstPtr& msg);

    /**
     * @brief Callback to the side-scan sonar ping data 
     * 
     * @param msg ROS Ping message
     */
    void pingCallback(const deepersense_msgs::Ping::ConstPtr& msg);

    /**
     * @brief Combined callback of side-scan sonar ping data and AUV navigation data
     * 
     * @param msg1 ROS Ping message 
     * @param msg2 ROS NavSts message
     */
    void combinedCallback(const deepersense_msgs::Ping::ConstPtr& msg1, 
                          const cola2_msgs::NavSts::ConstPtr& msg2);

    /**
     * @brief Publishes swath positions as a line marker on Rviz
     * 
     * @param start Swath starting point
     * @param finish Swath end point 
     */
    void addLineToMarker(const Eigen::Vector3d& start, 
                         const Eigen::Vector3d& finish);
    
    /**
     * @brief Publishes multiple swath positions as multiple line markers on Rviz
     * 
     * @param data Vector of start and end points for each swath 
     */
    void createLineMarkers(const std::vector<Eigen::Matrix<double, 6, 1> >& data);

    /**
     * @brief Publishes the robot's current position as a ROS Transform
     * 
     * @param position Robot position
     * @param orientation Robot orientation in radians (roll, pitch, yaw)
     */
    void publishRobotTF(const Eigen::Vector3f& position, 
                        const Eigen::Vector3f& orientation);

    /**
     * @brief Publishes the robot's current position as an Rviz Marker
     * 
     * @param marker Marker to publish 
     * @param position Robot position 
     * @param quat Robot orientation as quaternion 
     */
    void publishMarker(visualization_msgs::Marker& marker, 
                       const geometry_msgs::Vector3& position,
                       const tf2::Quaternion& quat);

};
