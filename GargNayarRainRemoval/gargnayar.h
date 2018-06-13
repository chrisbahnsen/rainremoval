//MIT License
//
//Copyright(c) 2018 Aalborg University
//Chris H. Bahnsen, June 2018
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files(the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions :
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

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
        candidatePixels,
        photometricConstraint,
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
	bool noGNProcessing;

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
