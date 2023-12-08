#pragma once 

#include <Eigen/Core>
#include <vector>
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>

#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>

#include <boost/foreach.hpp>
#include <boost/thread/mutex.hpp>

#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/exceptions.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf/tf.h>

#include <deepersense_msgs/Prediction.h>
#include <visualization_msgs/Marker.h>

namespace deepersense
{

/**
 * @brief 
 * 
 */
enum Action { STOP, SLOW_DOWN, GO_UP, GO_DOWN };

class Process 
{
public:
    /**
    * @brief Construct a new Process object
    * 
    * @param index Process index
    * @param action Action to perform 
    * @param startPos Current robot position
    * @param startOrient Current robot orientation 
    * @param frame Frame name
    * @param meshFile Robot mesh file path 
    * @param counter Process global counter 
    */
    Process(int index, 
            Action action, 
            const geometry_msgs::Vector3& startPos, 
            const geometry_msgs::Quaternion& startOrient,
            const std::string& frame,
            const std::string& meshFile,
            int counter)
    {
        index_ = index;
        action_ = action;
        startPos_ = startPos;
        startOrient_ = startOrient;
        step_ = 0;
        frame_ = frame;

        // create robot marker for visusalization 
        marker_.header.frame_id = "world";
        marker_.id = counter;
        marker_.ns = "";
        marker_.type = visualization_msgs::Marker::MESH_RESOURCE;
        marker_.mesh_use_embedded_materials = true;
        marker_.action = visualization_msgs::Marker::ADD;
        marker_.color.r = 0.0; marker_.color.g = 0.0; marker_.color.b = 0.0; marker_.color.a = 1.0;
        marker_.scale.x = 3.0; marker_.scale.y = 3.0; marker_.scale.z = 3.0;
        marker_.mesh_resource = meshFile;
        std::cout << marker_.mesh_resource << "\n";

        ROS_INFO("Created process %s", frame.c_str());
    }

    /**
     * @brief Index Get process index 
     * 
     * @return int Process index 
     */
    int index() { return index_; }

    /**
     * @brief Get step index
     * 
     * @return int Step index
     */
    int step() { return step_; }

    /**
     * @brief Get process action 
     * 
     * @return Action Process action
     */
    Action action() { return action_; }

    /**
     * @brief Get starting position
     * 
     * @return geometry_msgs::Vector3 Starting position
     */
    geometry_msgs::Vector3 startPos() { return startPos_; }

    /**
     * @brief Get starting orientation
     * 
     * @return geometry_msgs::Quaternion Starting orientation  
     */
    geometry_msgs::Quaternion startOrient() { return startOrient_; }
    
    /**
     * @brief Get previous position
     * 
     * @return geometry_msgs::Vector3 Previous position  
     */
    geometry_msgs::Vector3 prevPos() { return prevPos_; }

    /**
     * @brief Get previous orientation
     * 
     * @return geometry_msgs::Quaternion Previous orientation  
     */
    geometry_msgs::Quaternion prevOrient() { return prevOrient_; }

    /**
     * @brief Get velocity 
     * 
     * @return geometry_msgs::Vector3 Velocity
     */
    geometry_msgs::Vector3 velocity() { return velocity_; }

    /**
     * @brief Get frame name 
     * 
     * @return std::string Frame name 
     */
    std::string frame() { return frame_; }

    /**
     * @brief Set the previous position 
     * 
     * @param position Position 
     */
    void setPrevPos(const geometry_msgs::Vector3& position) { prevPos_ = position; }

    /**
     * @brief Set the previous orientation 
     * 
     * @param orientation Orientation
     */
    void setPrevOrient(const geometry_msgs::Quaternion& orientation) { prevOrient_ = orientation; }

    /**
     * @brief Set the velocity 
     * 
     * @param velocity Velocity
     */
    void setVelocity(const geometry_msgs::Vector3& velocity) { velocity_ = velocity; }

    /**
     * @brief Increment current step
     * 
     */
    void incrementStep() { ++step_; } 

    /**
     * @brief Decrease current process index 
     * 
     */
    void decreaseIndex() { --index_; }

    /**
     * @brief Publish robot pose marker 
     * 
     * @param position Position
     * @param orient Orientation
     * @param publisher ROS publisher 
     */
    void publishMarker(const geometry_msgs::Vector3& position,
                       const geometry_msgs::Quaternion& orient,
                       ros::Publisher& publisher,
                       Eigen::Vector3f rotationOffset = Eigen::Vector3f::Zero())
    {
        marker_.header.stamp = ros::Time();
        marker_.pose.position.x = position.x;
        marker_.pose.position.y = position.y;
        marker_.pose.position.z = position.z;

        double roll, pitch, yaw;
        tf2::Quaternion q(orient.x, orient.y, orient.z, orient.w);
        tf2::Matrix3x3 m(q);
        m.getRPY(roll, pitch, yaw);
        
        tf2::Quaternion q2;
        q2.setRPY(roll + rotationOffset(0), 
                  pitch + rotationOffset(1),
                  yaw + rotationOffset(2));

        marker_.pose.orientation.x = q2.x();
        marker_.pose.orientation.y = q2.y();
        marker_.pose.orientation.z = q2.z();
        marker_.pose.orientation.w = q2.w();
        publisher.publish(marker_);
    }

    /**
     * @brief Remove marker 
     * 
     * @param publisher Publisher 
     */
    void removeMarker(ros::Publisher& publisher)
    {
        marker_.action = visualization_msgs::Marker::DELETE;
        publisher.publish(marker_);
    }

private:
    int index_;
    int step_;
    Action action_;

    std::string frame_;

    geometry_msgs::Vector3 startPos_;
    geometry_msgs::Quaternion startOrient_;

    geometry_msgs::Vector3 prevPos_;
    geometry_msgs::Quaternion prevOrient_;

    geometry_msgs::Vector3 velocity_;

    visualization_msgs::Marker marker_;
};


class BehaviourChange
{
public:

    /**
    * @brief Construct a new Behaviour Change object
    * 
    * @param nh Node handle 
    * @param privateNh Private node handle 
    */
    BehaviourChange(ros::NodeHandle* nh, 
                    ros::NodeHandle* privateNh);

    /**
     * @brief Construct a new Behaviour Change object
     * 
     */
    BehaviourChange() { counter_[STOP] = counter_[SLOW_DOWN] = counter_[GO_UP] = counter_[GO_DOWN] = 0; }

    /**
     * @brief Adds change of behaviour to perform 
     * 
     * @param startPosition Current robot position  
     * @param startOrient Current robot orientation 
     * @param action Action to perform 
     */
    void addProcess(const geometry_msgs::Vector3& startPosition,
                    const geometry_msgs::Quaternion& startOrient,
                    Action action);

    /**
     * @brief Runs one step of all active processes 
     * 
     */
    void runProcesses();

private:

    /**
     * @brief Calculates current robot velocity and runs one step of all active processes 
     * 
     */
    void run();

    /**
     * @brief Perform one step of the process 
     * 
     * @param processIndex Index of the process
     */
    void performAction(int processIndex);

    /**
     * @brief Perform stop action for one step
     * 
     * @param index Index of the process
     */
    void stop(int index);

    /**
     * @brief Perform change in altitude action for one step
     * 
     * @param index Index of the process
     * @param up 
     */
    void changeAltitude(int index,
                        bool up);

    /**
     * @brief Perform slow down action for one step
     * 
     * @param index Index of the process
     */
    void slowDown(int index);

    /**
     * @brief Publish robot marker and transform
     * 
     * @param frame Frame name
     * @param pos Position 
     * @param orient Orientation 
     */
    void publishTransform(const std::string& frame, 
                          const geometry_msgs::Vector3& pos,
                          const geometry_msgs::Quaternion& orient);

    /**
     * @brief Model inference prediction callback
     * 
     * @param msg ROS Prediction message 
     */
    void predictionCallback(const deepersense_msgs::Prediction::ConstPtr& msg);

    /**
     * @brief Get the current robot transform 
     * 
     * @param transform Output transform
     */
    void getCurrentFrame(geometry_msgs::TransformStamped& transform);

    /**
     * @brief Generate frame name for new process
     * 
     * @param action Action to perform
     * @return std::string Output frame name 
     */
    std::string getCurrentFrameName(Action action);

    std::vector<Process> processes_;
    boost::mutex mutex_;

    std::shared_ptr<boost::thread> thread_;

    ros::Subscriber sub_;
    ros::NodeHandle nh_;
    ros::NodeHandle privateNh_;

    std::map<Action, int> counter_;

    int globalCounter_;

    Eigen::Vector3f previousPosition_;
    Eigen::Vector3f currentVelocity_;

    int numSteps_;
    int stepRate_;
    float minAltitude_;
    float maxAltitude_;
    float slowDownRate_;
    float minConfidence_;
    float changeConfidence_;

    int previousBiggestClass_;
    float previousConfidence_;

    std::string meshFile_;
    ros::Publisher markerPub_;

    Eigen::Vector3f meshRotationOffset_;
};

}
