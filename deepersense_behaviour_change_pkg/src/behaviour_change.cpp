#include "behaviour_change.hpp"

deepersense::BehaviourChange::BehaviourChange(ros::NodeHandle* nh, 
                                              ros::NodeHandle* privateNh) : nh_(*nh), 
                                                                            privateNh_(*privateNh), 
                                                                            previousConfidence_(-1.0), 
                                                                            globalCounter_(1), 
                                                                            previousBiggestClass_(-1)

{
    // initialise counters 
    counter_[STOP] = counter_[SLOW_DOWN] = counter_[GO_UP] = counter_[GO_DOWN] = 0;

    // load parameters from ROS Param Server 
    privateNh_.getParam("/behaviour/num_steps", numSteps_);
    privateNh_.getParam("/behaviour/step_rate", stepRate_);
    privateNh_.getParam("/behaviour/min_altitude", minAltitude_);
    privateNh_.getParam("/behaviour/max_altitude", maxAltitude_);
    privateNh_.getParam("/behaviour/slow_down_rate", slowDownRate_);
    privateNh_.getParam("/behaviour/min_confidence", minConfidence_);
    privateNh_.getParam("/behaviour/change_confidence", changeConfidence_);

    std::vector<double> rotationOffset;
    privateNh_.getParam("/simulator/robot_mesh_rotation_offset", rotationOffset);
    meshRotationOffset_(0) = rotationOffset[0] * M_PI / 180.0;
    meshRotationOffset_(1) = rotationOffset[1] * M_PI / 180.0;
    meshRotationOffset_(2) = rotationOffset[2] * M_PI / 180.0;

    std::string meshFilePath;
    privateNh_.getParam("/simulator/robot_dae_file", meshFilePath);
    meshFile_ = meshFilePath; 

    // initialise publisher and subscriber 
    markerPub_ = nh_.advertise<visualization_msgs::Marker>("/girona/behaviour_change", 200);
    sub_ = nh_.subscribe<deepersense_msgs::Prediction>("/prediction/output", 100, boost::bind(&BehaviourChange::predictionCallback, this, _1));

    // initialise thread 
    thread_.reset(new boost::thread(boost::bind(&BehaviourChange::run, this)));
}

void deepersense::BehaviourChange::run()
{
    ros::Rate rate(stepRate_);

    int count = 0;
    while (ros::ok())
    {        
        // get current Girona position
        geometry_msgs::TransformStamped transform;
        getCurrentFrame(transform);

        // if not first iteration, calculate velocity with previous one 
        if (count > 0)
        {
            Eigen::Vector3f currentPosition = Eigen::Vector3f(transform.transform.translation.x,
                                                               transform.transform.translation.y,
                                                               transform.transform.translation.z);
            currentVelocity_ = (currentPosition - previousPosition_) * stepRate_;
            previousPosition_ = currentPosition;
        }
        // if first iterations, store position 
        else
        {
            previousPosition_ = Eigen::Vector3f(transform.transform.translation.x,
                                                transform.transform.translation.y,
                                                transform.transform.translation.z);
            ++count;
        }
        
        // run one step of current processesn
        runProcesses();
        rate.sleep();
    }
}

std::string deepersense::BehaviourChange::getCurrentFrameName(Action action)
{
    std::string frame = "girona1000";

    // based on action, create new frame name
    switch(action)
    {
    case STOP:
        frame += "_stop_" + std::to_string(counter_[action]);
        ++counter_[action];
        ++globalCounter_;
        break;
    case GO_UP:
        frame += "_up_" + std::to_string(counter_[action]);
        ++counter_[action];
        ++globalCounter_;
        break;
    case GO_DOWN:
        frame += "_down_" + std::to_string(counter_[action]);
        ++counter_[action];
        ++globalCounter_;
        break;
    case SLOW_DOWN:
        frame += "_slow_" + std::to_string(counter_[action]);
        ++counter_[action];
        ++globalCounter_;
        break;
    default:
        break;
    }
    return frame;
}

void deepersense::BehaviourChange::addProcess(const geometry_msgs::Vector3& startPos,
                                              const geometry_msgs::Quaternion& startOrient,
                                              Action action)
{
    // add new process to vector protected by mutex 
    mutex_.lock();
    processes_.push_back(Process(processes_.size(), action, startPos, startOrient, getCurrentFrameName(action), meshFile_, globalCounter_));
    mutex_.unlock();
}

void deepersense::BehaviourChange::runProcesses()
{   
    mutex_.lock();

    std::vector<bool> finished(processes_.size(), false);
    BOOST_FOREACH(Process& process, processes_)
    {
        // perform one step of process, increment step counter 
        performAction(process.index());
        process.incrementStep();

        // if finished, set as finished and remove marker 
        if (process.step() >= numSteps_)
        {
            finished[process.index()] = true;       
            process.removeMarker(markerPub_);
        }
    }

    int i = 0;
    while (!std::all_of(finished.begin(), finished.end(), [](bool v) {return !v; }))
    {
        // if finished remove process 
        if (finished[i])
        {
            for (int j = i; j != processes_.size(); ++j)
                processes_[j].decreaseIndex(); 

            processes_.erase(processes_.begin() + i);
            finished.erase(finished.begin() + i);
        }
        else
            ++i;
    }

    mutex_.unlock();
}

void deepersense::BehaviourChange::performAction(int index)
{
    // based on type of process, perform specific action 
    switch(processes_[index].action())
    {
        case STOP:
            stop(index);
            break;
        case GO_UP:
            changeAltitude(index, true);
            break;
        case GO_DOWN:
            changeAltitude(index, false);
            break;
        case SLOW_DOWN:
            slowDown(index);
            break;
        default:
            break;
    }
}

void deepersense::BehaviourChange::stop(int index)
{   
    // publish transform and marker 

    publishTransform(processes_[index].frame(), 
                     processes_[index].startPos(), 
                     processes_[index].startOrient());

    processes_[index].publishMarker(processes_[index].startPos(), 
                                    processes_[index].startOrient(), 
                                    markerPub_,
                                    meshRotationOffset_);
}

void deepersense::BehaviourChange::changeAltitude(int index,
                                                  bool up)
{
    // check if movement up or down 
    float z;
    if (up)
        z = processes_[index].startPos().z + (maxAltitude_ - processes_[index].startPos().z) * (processes_[index].step() / float(numSteps_ - 1));
    else
        z = processes_[index].startPos().z + (minAltitude_ - processes_[index].startPos().z) * (processes_[index].step() / float(numSteps_ - 1));
    
    geometry_msgs::Vector3 position;
    position.x = processes_[index].startPos().x;
    position.y = processes_[index].startPos().y;
    position.z = z;

    // publish transform and marker 

    publishTransform(processes_[index].frame(), 
                     position, 
                     processes_[index].startOrient());

    processes_[index].publishMarker(position, 
                                    processes_[index].startOrient(), 
                                    markerPub_,
                                    meshRotationOffset_);
}

void deepersense::BehaviourChange::slowDown(int index)
{
    geometry_msgs::Vector3 position;

    // if step greater than 1 calculate position based on slow down rate of velocity 
    if (processes_[index].step() > 0)
    {
        position.x = processes_[index].prevPos().x + (slowDownRate_ * currentVelocity_.x() / stepRate_);
        position.y = processes_[index].prevPos().y + (slowDownRate_ * currentVelocity_.y() / stepRate_);
        position.z = processes_[index].prevPos().z + (slowDownRate_ * currentVelocity_.z() / stepRate_);
    }
    else 
    {
        position = processes_[index].startPos();
    }

    processes_[index].setPrevPos(position);
    processes_[index].setPrevOrient(processes_[index].startOrient());
    
    // publish transform and marker 

    publishTransform(processes_[index].frame(), 
                     position, 
                     processes_[index].startOrient());

    processes_[index].publishMarker(position, 
                                    processes_[index].startOrient(), 
                                    markerPub_,
                                    meshRotationOffset_);
}

void deepersense::BehaviourChange::predictionCallback(const deepersense_msgs::Prediction::ConstPtr& msg)
{
    // calculate average confidence 
    float confidenceMean = std::accumulate(msg->confidences.begin(), msg->confidences.end(), 0.0) / msg->confidences.size();

    // get unique classes 
    std::map<int, int> uniqueClasses;
    for (int i = 0; i != msg->outputs.size(); ++i)
    {
        int currentClass = (int) msg->outputs[i];
        if (uniqueClasses.count(currentClass))
            ++uniqueClasses[currentClass];
        else
            uniqueClasses.insert({currentClass, 0});
    }
    
    geometry_msgs::TransformStamped transform;
    
    // check class with biggest occupation 
    int biggestClass = -1;
    float biggestPercent = 0.0f;
    for(std::map<int, int>::iterator it = uniqueClasses.begin(); it != uniqueClasses.end(); ++it)
    {
        float percent = it->second / float(msg->outputs.size()); 
        if (percent > biggestPercent)
        {
            biggestPercent = percent;
            biggestClass = it->first;
        }
    }
    
    if (previousBiggestClass_ == -1)
        previousBiggestClass_ = biggestClass;
    else
    {
        if (confidenceMean <= minConfidence_)
        {
            ROS_WARN("Stop.");
            getCurrentFrame(transform);
            addProcess(transform.transform.translation, transform.transform.rotation, Action::STOP);   
        }
        else
        {
            // if biggest class changed from previous iteration, slow down robot 
            if (biggestClass != previousBiggestClass_)
            {
                ROS_WARN("Slow down.");
                
                getCurrentFrame(transform);
                addProcess(transform.transform.translation, transform.transform.rotation, Action::SLOW_DOWN);
                
                previousBiggestClass_ = biggestClass;
            }
        }
    }
}

void deepersense::BehaviourChange::getCurrentFrame(geometry_msgs::TransformStamped& transform)
{
    static tf2_ros::Buffer buffer;
    static tf2_ros::TransformListener listener(buffer);
    static ros::Rate rate(10.0);

    while (ros::ok())
    {
        try
        {
            transform = buffer.lookupTransform("world", "girona1000_original", ros::Time(0.0));
            break;
        }
        catch (tf2::TransformException& e) 
        {
            // ROS_ERROR("%s", e.what());
        }
        rate.sleep();
    }
}


void deepersense::BehaviourChange::publishTransform(const std::string& frame, 
                                                    const geometry_msgs::Vector3& pos,
                                                    const geometry_msgs::Quaternion& orient)
{
    //convert geometry_msgs to transform 
    static tf2_ros::TransformBroadcaster br;
    geometry_msgs::TransformStamped transformStamped;

    transformStamped.header.stamp = ros::Time::now();
    transformStamped.header.frame_id = "world";
    transformStamped.child_frame_id = frame;

    transformStamped.transform.translation = pos;
    transformStamped.transform.rotation = orient;
    
    br.sendTransform(transformStamped);
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "behaviour_change_node");
    ros::NodeHandle nh;
    ros::NodeHandle privateNh("~");
    deepersense::BehaviourChange bc(&nh, &privateNh);
    
    ros::spin();
}
