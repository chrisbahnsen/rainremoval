#ifndef GARGNAYAR_H
#define GARGNAYAR_H

#include <deque>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2\opencv.hpp>
#include <Windows.h>

/*! Used to keep track of the state of the rain algorithm when processing derained frames
    by using intermediate output of the algorithm
*/
struct GNMethod {
    enum e {
        fullMethod,
        fullMethodAlternativeST,
        candidatePixels,
        photometricConstraint,
        correlationMagnitudeAlternativeST,
        correlationMagnitude,
        overview,
    };
};

struct GNOptions {
    enum e {
        STZerothTimeLag,
        STVaryingTimeLag,
    };
};

struct Modality {
    enum e {
        grayscale,
        color
    };
};

struct GNRainParameters 
{
    // Grey scale threshold from which candidate rain pixels are generated
    int c; 

    // Maximum value of beta used for the enforcement of the geometric constraint
    double betaMax;

    // Neighbourhood in pixels used for computing the spatio-temporal correlation
    int neighborhoodSize;

    // Number of frames used for calculating the temporal correlation
    int numCorrelationFrames;

    // Minimum correlation value if a direction is to be considered
    std::vector<float> minDirectionalCorrelation;

    // Maximum spread of correlation values, in {param}*10 degrees
    std::vector<int> maxDirectionalSpread;

    // Number of frames to search for a replacement for a rainy pixel in the current image.
    // If the value is 1, the previous and next frame is searched
    int numFramesReplacement;

    // Options
    bool saveOverviewImg;
    bool saveDiffImg;
    bool useMedianBlur;
    bool verbose;

};

struct GNRainImage {
    std::vector<cv::Mat> prevRainImgs;
    std::vector<cv::Mat> nextRainImgs;
    cv::Mat currentRainImg;

    GNRainParameters customParams;

    std::string outputFolder;
};

class GargNayarRainRemover
{
public:
    GargNayarRainRemover(std::string inputVideo, std::string outputFolder, GNRainParameters rainParams = GargNayarRainRemover::getDefaultParameters());

    static GNRainParameters getDefaultParameters();
    static GNRainParameters loadParameters(std::string filePath);
    int saveParameters(std::string filePath);

    int removeRain();

private:
    void findCandidatePixels(cv::Mat prevImg, cv::Mat currentImg, cv::Mat nextImg, cv::Mat &outImg);
    void enforcePhotometricConstraint(cv::Mat inBinaryImg, cv::Mat prevImg, cv::Mat currentImg, cv::Mat &outImg);
    void computeSTCorrelation(std::deque<cv::Mat> binaryFieldHistory, cv::Mat &outImg, int method);

    void computeCorrelationDirection(cv::Mat inImg, cv::Mat &outImg, float minDirectionalCorrelation, int maxDirectionalSpread);
    void computeDirectionalMasks();

    void removeDetectedRain(std::vector<cv::Mat> prevImgs, cv::Mat currentImg, std::vector<cv::Mat> nextImgs,
        GNRainImage rainImage, cv::Mat &rainRemovedCurrentImg);
    int fetchNextImage(cv::VideoCapture &cap, std::map<int, std::vector<cv::Mat> > &prevImgs, std::map<int, cv::Mat> &currentImg, std::map<int, std::vector<cv::Mat> > &nextImgs);

    std::string inputVideo, outputFolder;
    std::vector<cv::Mat> directionalMasks; // According to the OpenCV documentation, use Mat for small templates

    GNRainParameters rainParams;
};

#endif // !GARGNAYAR_H
