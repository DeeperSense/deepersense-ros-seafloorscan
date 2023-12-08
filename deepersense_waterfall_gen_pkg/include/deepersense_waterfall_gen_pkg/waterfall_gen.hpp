#pragma once 

#include <Eigen/Core>
#include <Eigen/Dense>

#include <deque>

#include "waterfall_utils.hpp"

namespace deepersense
{

class SynthesizeWaterfall
{
public:
  
    /**
     * @brief Construct a new Synthesize Waterfall object
     * 
     */
    SynthesizeWaterfall(): pingNumber_(0), init_(false) {}

    /**
     * @brief Destroy the SynthesizeWaterfall object
     * 
     */
    ~SynthesizeWaterfall() {}

    /**
     * @brief Calculate bin indices and bin indices from centre bin 
     * 
     */
    void loadBinsInfo();

    /**
     * @brief Get maximum number of pings per waterfall
     * 
     * @return int& Maximum number of pings per waterfall 
     */
    int& maxPings() { return maxPings_; }

    /**
     * @brief Get slant resolution 
     * 
     * @return float& Slant resolution
     */
    float& slantRes() { return slantRes_; }

    /**
     * @brief Get slant range
     * 
     * @return float& Slant range
     */
    float& slantRange() { return slantRange_; }

    /**
     * @brief Get number of samples 
     * 
     * @return int& Number of samples
     */
    int& numberSamples() { return numSamples_; }

    /**
     * @brief Get interpolation window 
     * 
     * @return int& Interpolation window
     */
    int& window() { return window_; }

    
    /**
     * @brief Load swath positions into a vector
     * 
     * @param swaths Vector of swath positions 
     */
    void copySwaths(std::vector<Eigen::Vector2d>& swaths) { swaths = std::vector<Eigen::Vector2d>({swaths_.begin(), swaths_.end()}); }
    
    /**
     * @brief Load swath data into a vector 
     * 
     * @param swathData Vector of swath data 
     */
    void copySwathData(std::vector<Eigen::Vector4d>& swathData) { swathData = std::vector<Eigen::Vector4d>({swathData_.begin(), swathData_.end()});}

    /**
     * @brief Whether to process data inside queues or not 
     * 
     * @return true Process data if conditions satisfied 
     * @return false Do not process data if conditions not satisfied
     */
    bool interpolateData() { return (!init_ && (pingNumber_ >= maxPings_)) || (init_ && (pingNumber_ >= window_)); }

    /**
     * @brief Process ping data to obtain the swath positions  
     * 
     * @param yaw Yaw in radians
     * @param east UTM x coordinate
     * @param north UTM y coordinate 
     * @param altitude Altitude in metres 
     * @param pitch Pitch in radians
     * @param intensities Vector of float intensities 
     * @param swath Output swath 
     */
    void processPing(float yaw, 
                     double east, 
                     double north, 
                     float altitude, 
                     float pitch, 
                     std::vector<int> intensities, 
                     std::vector<Eigen::Vector2d>& swath);

    /**
     * @brief Get the ping number 
     * 
     * @return int Ping number
     */
    int getPingNumber() { return pingNumber_; }
    
    /**
     * @brief Process accumulated data 
     * 
     * @param correctedLines Corrected swath positions 
     * @param originalIntensities Original intensity values 
     * @param correctedIntensities Intensity values after correction
     * @param numGroundBins Number of non water-column bins 
     * @param numBlindBins Number of bins in water-column
     * @param correctWaterfall Whether to apply waterfall correction 
     */
    void interpolate(std::vector<Eigen::Vector2d>& correctedLines,
                     std::vector<int>& originalIntensities,
                     std::vector<int>& correctedIntensities,
                     int numGroundBins,
                     int numBlindBins,
                     bool correctWaterfall);

private:

    int maxPings_;
    int window_;
    float slantRange_;

    int numSamples_;
    float slantRes_;

    bool init_;

    std::vector<int> bins_;
    std::vector<float> binsFromCentre_;
    float binCentral_;

    std::deque<int> intensities_;
    std::deque<Eigen::Vector2d> swaths_;
    std::deque<Eigen::Vector4d> swathData_;

    int pingNumber_;
};


}
