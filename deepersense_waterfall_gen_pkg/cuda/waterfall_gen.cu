#include "waterfall_gen.hpp"

namespace deepersense {

void SynthesizeWaterfall::loadBinsInfo()
{
    slantRes_ = slantRange_ * 2 / numSamples_;
    binCentral_ = (numSamples_ - 1) / 2.0;

    for (int i = 0; i != numSamples_; ++i)
    {
        bins_.push_back(i);
        binsFromCentre_.push_back(float(i) - binCentral_);
    }
}

void SynthesizeWaterfall::processPing(float yaw, 
                                      double east, 
                                      double north, 
                                      float altitude, 
                                      float pitch,
                                      std::vector<int> intensities,
                                      std::vector<Eigen::Vector2d>& swath)
{
    // calculate swath positions 
    swath.reserve(numSamples_);
    float realAltitude = altitude / std::cos(pitch);
    
    utils::createSwath(yaw, realAltitude, east, north, slantRes_, numSamples_, bins_, binsFromCentre_, swath);

    // add new data to queues 
    for (int i = 0; i != numSamples_; ++i)
    {
        swaths_.emplace_back(swath[i]);
        intensities_.emplace_back(intensities[i]);
    }

    swathData_.emplace_back(Eigen::Vector4d(east, north, altitude, yaw));
    pingNumber_ += 1;
}

void SynthesizeWaterfall::interpolate(std::vector<Eigen::Vector2d>& correctedLines,
                                      std::vector<int>& originalIntensities,
                                      std::vector<int>& correctedIntensities,
                                      int numGroundBins,
                                      int numBlindBins,
                                      bool correctWaterfall)
{
    pingNumber_ = 0;

    if (!init_)
        init_ = true;

    // remove past data from queues 
    while (swaths_.size() > numSamples_ * maxPings_)
    {
        swaths_.pop_front();
        intensities_.pop_front();

        if (swathData_.size() > maxPings_)
            swathData_.pop_front();
    }

    // load intensity data into vector
    originalIntensities = std::vector<int>({intensities_.begin(), intensities_.end()});

    // check whether waterfall correction is needed 
    if (correctWaterfall)
    {
        // load swath data into a vector 
        correctedLines.resize(maxPings_ * numSamples_);
        std::vector<Eigen::Vector4d> swathData({swathData_.begin(), swathData_.end()});

        correctedIntensities.resize(maxPings_ * numSamples_);

        // watercolumn mask 
        std::vector<int> mask(numSamples_ * maxPings_, 1);
        for (int i = 0; i != maxPings_; ++i){
            for (int j = 0; j != numSamples_; ++j)
            {
                if (j >= numGroundBins && j <= numGroundBins + 2 * numBlindBins)
                    mask[i*numSamples_+j] = 0;
            }
        }

        // load swath positions into a vector 
        std::vector<Eigen::Vector2d> inputLines({swaths_.begin(), swaths_.end()});

        int maxNeighbours = 8;
        float resolution = 0.05;

        utils::interpolate(maxPings_, numSamples_, resolution, slantRes_, maxNeighbours, 
                            bins_, binsFromCentre_, swathData, inputLines, 
                            originalIntensities, mask, correctedLines, correctedIntensities);
    }
}   

}
