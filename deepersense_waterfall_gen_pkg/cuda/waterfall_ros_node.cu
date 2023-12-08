#include "waterfall_ros_node.hpp"


WaterfallNode::WaterfallNode(ros::NodeHandle* nh, 
                             ros::NodeHandle* privateNh): nh_(*nh), 
                                                          privateNh_(*privateNh), 
                                                          imageCount_(0)

{
    // load parameters from ROS Param Server 

    privateNh_.getParam("/simulator/pings_topic_name", pingsTopic_);
    privateNh_.getParam("/simulator/ping_info_topic_name", pingInfoTopic_);
    privateNh_.getParam("/simulator/nav_topic_name", navTopic_);
    privateNh_.getParam("/simulator/separate_inputs", separateInputs_);

    privateNh_.getParam("/undistortion/correct_waterfall", correctWaterfall_);
    privateNh_.getParam("/visualisation/publish_swath_lines", visSwathLines_);

    std::vector<double> rotationOffset;
    privateNh_.getParam("/simulator/robot_mesh_rotation_offset", rotationOffset);
    meshRotationOffset_(0) = rotationOffset[0] * M_PI / 180.0;
    meshRotationOffset_(1) = rotationOffset[1] * M_PI / 180.0;
    meshRotationOffset_(2) = rotationOffset[2] * M_PI / 180.0;

    float slantRange;
    int numSamples, maxPings, interpolationWindow;
    privateNh_.getParam("/sonar/slant_range", slantRange);
    privateNh_.getParam("/sonar/num_samples", numSamples);
    privateNh_.getParam("/undistortion/max_pings", maxPings);
    privateNh_.getParam("/undistortion/interpolation_window", interpolationWindow);

    waterfallGen_.maxPings() = maxPings;
    waterfallGen_.window() = interpolationWindow;
    waterfallGen_.slantRange() = slantRange;
    waterfallGen_.numberSamples() = numSamples;

    ROS_INFO("Number of pings per waterfall: %d", maxPings);
    ROS_INFO("Interpolation window: %d", interpolationWindow);
    ROS_INFO("Slant range: %f", slantRange);

    pingInfoSub_ = nh_.subscribe<deepersense_msgs::PingInfo>(pingInfoTopic_, 1, boost::bind(&WaterfallNode::pingInfoCallback, this, _1));
    patchPub_ = nh_.advertise<deepersense_msgs::Patch>("/patches", maxPings);
    markerPub_ = nh_.advertise<visualization_msgs::Marker>("/girona/original_path", maxPings);

    pingCombinedSub_.subscribe(nh_, pingsTopic_, 100);
    navCombinedSub_.subscribe(nh_, navTopic_, 100);

    // create robot marker for visualization 
    std::string urdfDir, meshFilePath;
    privateNh_.getParam(ros::this_node::getName() + "/urdf_directory", urdfDir);
    privateNh_.getParam("/simulator/robot_dae_file", meshFilePath);
    robotMarker_.header.frame_id = "world";
    robotMarker_.id = 0;
    robotMarker_.ns = "";
    robotMarker_.type = visualization_msgs::Marker::MESH_RESOURCE;
    robotMarker_.mesh_use_embedded_materials = true;
    robotMarker_.action = visualization_msgs::Marker::ADD;
    robotMarker_.color.r = 0.0; robotMarker_.color.g = 0.0; robotMarker_.color.b = 0.0; robotMarker_.color.a = 1.0;
    robotMarker_.scale.x = 3.0; robotMarker_.scale.y = 3.0; robotMarker_.scale.z = 3.0;
    robotMarker_.mesh_resource = meshFilePath; //"file://" + urdfDir + meshFilePath;
    
    // Visualisation 
    if (visSwathLines_)
    {
        rawSwathsPub_ = nh_.advertise<visualization_msgs::Marker>("/uncorrected_swaths", maxPings);
        correctedSwathsPub_ = nh_.advertise<visualization_msgs::Marker>("/corrected_swaths", maxPings);
        
        rawSwaths_.id = 0;
        rawSwaths_.type = visualization_msgs::Marker::LINE_LIST;
        rawSwaths_.scale.x = 0.01;
        rawSwaths_.color.r = 1.0;
        rawSwaths_.color.a = 1.0;
        rawSwaths_.action = visualization_msgs::Marker::ADD;

        rawSwaths_.pose.orientation.x = 0.0;
        rawSwaths_.pose.orientation.y = 0.0;
        rawSwaths_.pose.orientation.z = 0.0;
        rawSwaths_.pose.orientation.w = 1.0;

        correctedSwaths_.id = 1;
        correctedSwaths_.type = visualization_msgs::Marker::LINE_LIST;
        correctedSwaths_.scale.x = 0.01;
        correctedSwaths_.color.b = 1.0;
        correctedSwaths_.color.a = 1.0;
        correctedSwaths_.action = visualization_msgs::Marker::ADD;

        correctedSwaths_.pose.orientation.x = 0.0;
        correctedSwaths_.pose.orientation.y = 0.0;
        correctedSwaths_.pose.orientation.z = 0.0;
        correctedSwaths_.pose.orientation.w = 1.0;
    }
}

void WaterfallNode::pingInfoCallback(const deepersense_msgs::PingInfo::ConstPtr& msg)
{
    if (msg->slant_range != waterfallGen_.slantRange())
    {
        ROS_ERROR("Slant range from the XTF file (%f) does not match the one in the configuration file (%f).", 
            msg->slant_range, waterfallGen_.slantRange());
        ros::shutdown();
    }

    if (msg->num_samples != waterfallGen_.numberSamples())
    {
        ROS_ERROR("Number of samples from the XTF file (%f) does not match the one in the configuration file (%f).",
            msg->num_samples, waterfallGen_.numberSamples());
        ros::shutdown();
    }
    waterfallGen_.loadBinsInfo();

    ROS_INFO("Ping info initialised.");

    // depending on the number of input topics, create callback(s)
    if (separateInputs_)
    {
        sync_.reset(new Sync(SyncPolicy(500), pingCombinedSub_, navCombinedSub_));
        sync_->registerCallback(boost::bind(&WaterfallNode::combinedCallback, this, _1, _2));
    }
    else
        pingSub_ = nh_.subscribe<deepersense_msgs::Ping>(pingsTopic_, waterfallGen_.maxPings(), boost::bind(&WaterfallNode::pingCallback, this, _1));
    pingInfoSub_.shutdown();
}

void WaterfallNode::combinedCallback(const deepersense_msgs::Ping::ConstPtr& msg1, 
                                     const cola2_msgs::NavSts::ConstPtr& msg2)
{   
    double x = msg2->position.east;  
    double y = msg2->position.north;  
    double z = msg2->altitude; 
    
    float roll = msg2->orientation.roll;
    float pitch = msg2->orientation.pitch;
    float yaw = msg2->orientation.yaw; // -1 

    std::vector<int> intensities = msg1->intensities;

    onPingCallback(x, y, z, roll, pitch, yaw, intensities);
}

void WaterfallNode::pingCallback(const deepersense_msgs::Ping::ConstPtr& msg)
{
    double x = msg->x;
    double y = msg->y;
    double z = msg->altitude;
    std::vector<int> intensities = msg->intensities;

    float roll = msg->roll;
    float pitch = msg->pitch;
    float yaw = msg->yaw; // -1 

    onPingCallback(x, y, z, roll, pitch, yaw, intensities);
}

void WaterfallNode::onPingCallback(double x, 
                                   double y, 
                                   double altitude, 
                                   float roll, 
                                   float pitch, 
                                   float yaw, 
                                   const std::vector<int>& intensities)
{
    // store initial positions to create a local coordinate system 
    static Eigen::Vector3d initialPosition = Eigen::Vector3d(x, y, altitude);
    static float maxAltitude = std::numeric_limits<float>::min();
    static float initialYaw = yaw;

    // update maximum altitude 
    if (altitude > maxAltitude)
        maxAltitude = altitude;
    
    // process data 
    std::vector<Eigen::Vector2d> swath;
    waterfallGen_.processPing(yaw, x, y, altitude, pitch, intensities, swath); 

    // visualisation
    publishRobotTF(Eigen::Vector3f(x, y, altitude), Eigen::Vector3f(roll, pitch, yaw));
    if (visSwathLines_)
        addLineToMarker(Eigen::Vector3d(swath[0].x(), swath[0].y(), altitude),
                    Eigen::Vector3d(swath[waterfallGen_.numberSamples()-1].x(), 
                    swath[waterfallGen_.numberSamples()-1].y(), altitude));
  
    if (waterfallGen_.interpolateData())
    {
        ROS_INFO("Interpolate data.");

        int numBlindBins = (int) std::round(maxAltitude / waterfallGen_.slantRes());
        int numGroundBins = (int) waterfallGen_.numberSamples() / 2 - numBlindBins;
        
        // correct waterfall 
        std::vector<Eigen::Vector2d> correctedLines;
        std::vector<int> intensitiesOld;
        std::vector<int> intensitiesNew;
        waterfallGen_.interpolate(correctedLines, intensitiesOld, intensitiesNew, numGroundBins, numBlindBins, correctWaterfall_);
	
        // publish waterfall 
        deepersense_msgs::Patch patch;
        patch.height = waterfallGen_.maxPings();
        patch.width = waterfallGen_.numberSamples();
        
        patch.distorted = intensitiesOld;
        if (correctWaterfall_)
            patch.undistorted = intensitiesNew;            

        std::vector<Eigen::Vector4d> swathData;
        std::vector<Eigen::Vector2d> swaths;
        waterfallGen_.copySwathData(swathData);
        waterfallGen_.copySwaths(swaths);
        std::vector<geometry_msgs::Point> positions;
       
        // store local positions 
        for (int i = 0; i != swaths.size(); ++i)
        {
            int idx = i / waterfallGen_.numberSamples();
            geometry_msgs::Point point;
            if (correctWaterfall_)
            {
                point.x = correctedLines[i].x() - initialPosition(0);
                point.y = correctedLines[i].y() - initialPosition(1);
                point.z = swathData[idx].z() - initialPosition(2);
            }
            else
            {
                point.x = swaths[i].x() - initialPosition(0);
                point.y = swaths[i].y() - initialPosition(1);
                point.z = swathData[idx].z() - initialPosition(2);
            }
            positions.push_back(point);
        }
       
        // publish data for model inference 
        patch.blind_bins = numBlindBins;
        patch.ground_bins = numGroundBins;
        patch.positions = positions;
        patchPub_.publish(patch);

        if (visSwathLines_ && correctWaterfall_)
        {
            // plot lines
            std::vector<Eigen::Matrix<double, 6, 1> > correctedLinesEndPoints;
            for (int i = 0; i != waterfallGen_.maxPings(); ++i)
            {
                Eigen::Matrix<double, 6, 1> data;
                data << correctedLines[i*waterfallGen_.numberSamples()](0), 
                        correctedLines[i*waterfallGen_.numberSamples()](1),
                        swathData[i](2),
                        correctedLines[i*waterfallGen_.numberSamples() + waterfallGen_.numberSamples() - 1](0), 
                        correctedLines[i*waterfallGen_.numberSamples() + waterfallGen_.numberSamples() - 1](1),
                        swathData[i](2);
                correctedLinesEndPoints.push_back(data);
            }
            createLineMarkers(correctedLinesEndPoints);
        }

        // reset max altitude
        maxAltitude = std::numeric_limits<float>::min();
    }
}

void WaterfallNode::addLineToMarker(const Eigen::Vector3d& start, 
                                    const Eigen::Vector3d& finish)
{ 
    static Eigen::Vector3d initPosition = (start + finish) / 2.0;

    geometry_msgs::Point p1;
    p1.x = start(0) - initPosition(0);
    p1.y = start(1) - initPosition(1);
    p1.z = 0.0;

    geometry_msgs::Point p2;
    p2.x = finish(0) - initPosition(0);
    p2.y = finish(1) - initPosition(1);
    p2.z = 0.0;

    rawSwaths_.header.frame_id = "world";
    rawSwaths_.header.stamp = ros::Time::now();

    rawSwaths_.points.push_back(p1);
    rawSwaths_.points.push_back(p2);

    rawSwathsPub_.publish(rawSwaths_);
}

void WaterfallNode::createLineMarkers(const std::vector<Eigen::Matrix<double, 6, 1> >& data)
{
    static Eigen::Vector3d initPosition = (data[0].block<3,1>(0,0) + data[0].block<3,1>(3,0)) / 2.0;

    for (int i = 0; i != data.size(); ++i)
    {
        geometry_msgs::Point p1;
        p1.x = data[i](0) - initPosition(0);
        p1.y = data[i](1) - initPosition(1);
        p1.z = data[i](2) - initPosition(2);

        geometry_msgs::Point p2;
        p2.x = data[i](3) - initPosition(0);
        p2.y = data[i](4) - initPosition(1);
        p2.z = data[i](5) - initPosition(2);

        correctedSwaths_.header.frame_id = "world";
        correctedSwaths_.header.stamp = ros::Time::now();

        correctedSwaths_.points.push_back(p1);
        correctedSwaths_.points.push_back(p2);
    }

    correctedSwathsPub_.publish(correctedSwaths_);
}

void WaterfallNode::publishMarker(visualization_msgs::Marker& marker, 
                                  const geometry_msgs::Vector3& position,
                                  const tf2::Quaternion& quat)
{
    marker.header.stamp = ros::Time();
    marker.pose.position.x = position.x;
    marker.pose.position.y = position.y;
    marker.pose.position.z = position.z;

    marker.pose.orientation.x = quat.x();
    marker.pose.orientation.y = quat.y();
    marker.pose.orientation.z = quat.z();
    marker.pose.orientation.w = quat.w();
    markerPub_.publish(marker);
}

void WaterfallNode::publishRobotTF(const Eigen::Vector3f& position, 
                                   const Eigen::Vector3f& orientation)
{
    static tf2_ros::TransformBroadcaster br;
    geometry_msgs::TransformStamped transformStamped;

    transformStamped.header.stamp = ros::Time::now();
    transformStamped.header.frame_id = "world";
    transformStamped.child_frame_id = "girona1000_original";

    static Eigen::Vector3f initPosition = position;

    transformStamped.transform.translation.x = position(0) - initPosition(0);
    transformStamped.transform.translation.y = position(1) - initPosition(1);
    transformStamped.transform.translation.z = position(2);

    tf2::Quaternion q;
    q.setRPY(orientation(0), orientation(1), orientation(2));
    transformStamped.transform.rotation.x = q.x();
    transformStamped.transform.rotation.y = q.y();
    transformStamped.transform.rotation.z = q.z();
    transformStamped.transform.rotation.w = q.w();

    br.sendTransform(transformStamped);

    tf2::Quaternion q2;
    q2.setRPY(orientation(0) + meshRotationOffset_(0), 
              orientation(1) + meshRotationOffset_(1),
              orientation(2) + meshRotationOffset_(2));
    publishMarker(robotMarker_, transformStamped.transform.translation, q2);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "waterfall_preprocess_node");

    ros::NodeHandle nh;
    ros::NodeHandle privateNh("~");

    WaterfallNode node(&nh, &privateNh);

    ros::spin();

    return 0;
}
