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


#include "gargnayar.h"

using namespace std;
using namespace cv;





GargNayarRainRemover::GargNayarRainRemover(std::string inputVideo, std::string outputFolder, GNRainParameters rainParams)
{
    this->inputVideo = inputVideo;
    this->outputFolder = outputFolder;
    this->rainParams = rainParams;

    computeDirectionalMasks(); // Initialise the directional masks
}

//! Retrieves default parameters as used in the article "Rain Removal in Traffic Surveillance: Does it Matter?"
/*!
\return struct of default parameters

*/
GNRainParameters GargNayarRainRemover::getDefaultParameters()
{
    GNRainParameters defaultParams;

    defaultParams.c = 3;
    defaultParams.betaMax = 0.039;
    defaultParams.neighborhoodSize = 11;
    defaultParams.numCorrelationFrames = 30;
    defaultParams.minDirectionalCorrelation.push_back(0.40);
    defaultParams.maxDirectionalSpread.push_back(3);
    defaultParams.numFramesReplacement = 2;

    defaultParams.saveDiffImg = false;
    defaultParams.saveOverviewImg = false;
    defaultParams.useMedianBlur = false;
	defaultParams.noGNProcessing = false;

    return defaultParams;
}

//! Loads parameters saved in a OpenCV FileStorage compatible format
/*!
\param filePath load the parameters from this path, either relative or full
\return struct containing the loaded parameters

*/
GNRainParameters GargNayarRainRemover::loadParameters(std::string filePath)
{
    GNRainParameters newParams = getDefaultParameters();
    FileStorage fs(filePath, FileStorage::READ);

    if (fs.isOpened()) {
        int tmpInt;
        fs["c"] >> tmpInt;
        if (tmpInt != 0) {
            newParams.c = tmpInt;
        }

        double tmpDouble;
        fs["betaMax"] >> tmpDouble;
        if (tmpDouble != 0.) {
            newParams.betaMax = tmpDouble;
        }

        fs["neighborhoodSize"] >> tmpInt;
        if (tmpInt != 0) {
            newParams.neighborhoodSize = tmpInt;
        }
            
        fs["numCorrelationFrames"] >> tmpInt;
        if (tmpInt != 0) {
            newParams.numCorrelationFrames = tmpInt;
        }
         
        vector<float> tmpFloatVec;
        fs["minDirectionalCorrelation"] >> tmpFloatVec;
        if (!tmpFloatVec.empty()) {
            newParams.minDirectionalCorrelation = tmpFloatVec;
        }

        vector<int> tmpIntVec;
        fs["maxDirectionalSpread"] >> tmpIntVec;
        if (!tmpIntVec.empty()) {
            newParams.maxDirectionalSpread = tmpIntVec;
        }

        fs["numFramesReplacement"] >> tmpInt;
        if (tmpInt != 0) {
            newParams.numFramesReplacement = tmpInt;
        }

        fs["saveOverviewImg"] >> newParams.saveOverviewImg;
        fs["saveDiffImg"] >> newParams.saveDiffImg;
        fs["useMedianBlur"] >> newParams.useMedianBlur;
        fs["verbose"] >> newParams.verbose;
    }
    
    return newParams;
}

//! Save current parameters in a OpenCV FileStorage compatible format
/*!
\param filePath save the parameters to this path, either relative or full
\return 0 if the operation was sucessfull, 1 otherwise

*/
int GargNayarRainRemover::saveParameters(std::string filePath)
{
    FileStorage fs(filePath, FileStorage::WRITE);

    if (fs.isOpened()) {
        fs << "c" << rainParams.c;
        fs << "betaMax" << rainParams.betaMax;
        fs << "neighborhoodSize" << rainParams.neighborhoodSize;
        fs << "numCorrelationFrames" << rainParams.numCorrelationFrames;
        fs << "minDirectionalCorrelation" << rainParams.minDirectionalCorrelation;
        fs << "maxDirectionalSpread" << rainParams.maxDirectionalSpread;
        fs << "numFramesReplacement" << rainParams.numFramesReplacement;

        fs << "saveOverviewImg" << rainParams.saveOverviewImg;
        fs << "saveDiffImg" << rainParams.saveDiffImg;
        fs << "useMedianBlur" << rainParams.useMedianBlur;
        fs << "verbose" << rainParams.verbose;

		return 0;
    }
    else {
        return 1;
    }
}

//! Retrieves next image from video and handles bookkeeping of current and previous frames
/*!
    \param cap handle to opened VideoCapture container
    \param prevImgs vector of previous images, containing at least four elements
    \param currentImg matrix returned as the "current", time-shifted image
    \param nextImgs vector of "next" images. New frames from the video capture are placed here, and propagated through the current image and previous images
    \return if return == 0, the frame was retrieved successfully. If return == 1, no frame was retrieved

*/
int GargNayarRainRemover::fetchNextImage(cv::VideoCapture &cap, std::map<int, std::vector<cv::Mat> > &prevImgs, 
    std::map<int, cv::Mat> &currentImg, std::map<int, std::vector<cv::Mat> > &nextImgs)
{
    // Handle bookkeeping of frames
    prevImgs[Modality::color][3] = prevImgs[Modality::color][2].clone();
    prevImgs[Modality::grayscale][3] = prevImgs[Modality::grayscale][2].clone();
    prevImgs[Modality::color][2] = prevImgs[Modality::color][1].clone();
    prevImgs[Modality::grayscale][2] = prevImgs[Modality::grayscale][1].clone();
    prevImgs[Modality::color][1] = prevImgs[Modality::color][0].clone();
    prevImgs[Modality::grayscale][1] = prevImgs[Modality::grayscale][0].clone();

    prevImgs[Modality::color][0] = currentImg[Modality::color].clone();
    prevImgs[Modality::grayscale][0] = currentImg[Modality::grayscale].clone();
    currentImg[Modality::color] = nextImgs[Modality::color][0].clone();
    currentImg[Modality::grayscale] = nextImgs[Modality::grayscale][0].clone();

    nextImgs[Modality::color][0] = nextImgs[Modality::color][1].clone();
    nextImgs[Modality::grayscale][0] = nextImgs[Modality::grayscale][1].clone();

    nextImgs[Modality::color][1] = nextImgs[Modality::color][2].clone();
    nextImgs[Modality::grayscale][1] = nextImgs[Modality::grayscale][2].clone();

    cap >> nextImgs[Modality::color][2];

    if (nextImgs[Modality::color][2].empty()) {
        return 1;
    }

    cv::cvtColor(nextImgs[Modality::color][2], nextImgs[Modality::grayscale][2], COLOR_BGR2GRAY);

    return 0;
}

//! The main processing loop of the rain removal algorithm. Operates on the parameters given in rainParams of the class
int GargNayarRainRemover::removeRain()
{
    // Open video
    VideoCapture cap(inputVideo);

    map<int, vector<GNRainImage>> rainImgs;

    for (int method = GNMethod::fullMethod; method <= GNMethod::overview; ++method) {
        vector<GNRainImage> gnImages;

        string basePath;
        switch (method)
        {
        case GNMethod::fullMethod:
        {
			// The full method as described in Garg and Nayar in the paper 
			// "Detection and Removal of Rain from Videos"
            basePath = outputFolder + "/Full";
            break;
        }
        case GNMethod::candidatePixels:
        {
			// We consider all the candidate rain pixels to be rain pixels. 
			// Conforms to step (a) of Figure 6 in page 5 in the paper
            basePath = outputFolder + "/Candidate/";
            break;
        }
        case GNMethod::photometricConstraint:
        {
			// We apply the photometric constraint to the candidate pixels
			// Conforms to step (a) + (b) of Figure 6.
            basePath = outputFolder + "/Photometric/";
            break;
        }
        case GNMethod::correlationMagnitude:
        {
			// We apply the spatio-temporal correlation magnitude on top of a
			// collection of detected rain streaks found by the photometric constraint
			// Conforms to step (a) + (b) + (c) + (d) of Figure 6. 
			// If we add step (e), the direction of correlation, we will get the full 
			// method
            basePath = outputFolder + "/STCorr/";
            break;
        }
        case GNMethod::overview:
        {
			// Produces a neat overview image of the computational steps listed above.
            basePath = outputFolder + "/Overview/";
            break;
        }
        }
    

        if (method == GNMethod::fullMethod) {
            // Insert (possibly) different range of rain parameters in the rainImage struct

            for (auto minDirCorr : rainParams.minDirectionalCorrelation) {
                for (auto maxDirSpread : rainParams.maxDirectionalSpread) {
                    GNRainImage gnImage;
                    gnImage.prevRainImgs.resize(2);
                    gnImage.nextRainImgs.resize(2);
                    gnImage.customParams = rainParams;

                    gnImage.customParams.minDirectionalCorrelation.clear();
                    gnImage.customParams.maxDirectionalSpread.clear();

                    gnImage.customParams.minDirectionalCorrelation.push_back(minDirCorr);
                    gnImage.customParams.maxDirectionalSpread.push_back(maxDirSpread);

                    stringstream ssMinDirCorr, maxDir;
                    ssMinDirCorr << setw(3) << setfill('0') << std::setprecision(2) << minDirCorr;

                    gnImage.outputFolder = basePath + "dirCorr-" + ssMinDirCorr.str()+
                        "-maxDir-" + to_string(maxDirSpread) + "/";

                    gnImages.push_back(gnImage);
                }
            }

        }
        else {
            // Insert the modified rainImage with default parameters. Only one image struct is needed
            GNRainImage gnImage;
            gnImage.prevRainImgs.resize(2);
            gnImage.nextRainImgs.resize(2);
            gnImage.customParams = rainParams;
            gnImage.outputFolder = basePath;
            gnImages.push_back(gnImage);
        }

        rainImgs[method] = gnImages;
    }


    for (int method = GNMethod::fullMethod; method <= GNMethod::overview; ++method) {
        for (auto&& param : rainImgs[method]) {
            CreateDirectoryA(param.outputFolder.c_str(), NULL);
        }
    }  

    if (rainParams.useMedianBlur) {
        string dir = outputFolder + "/Median/";
        CreateDirectoryA(dir.c_str(), NULL);
    }
    
    if (cap.isOpened()) {
        cout << "Opening video: " << inputVideo << endl;

        // Extract the first five frames
        map<int, Mat> currentImg;
        map<int, vector<Mat>> prevImgs, nextImgs;
        prevImgs[Modality::grayscale].resize(4);
        prevImgs[Modality::color].resize(4);
        nextImgs[Modality::grayscale].resize(3);
        nextImgs[Modality::color].resize(3);

        // Convert images to greyscale
        cap >> prevImgs[Modality::color][2]; cv::cvtColor(prevImgs[Modality::color][2], prevImgs[Modality::grayscale][2], COLOR_BGR2GRAY);
        cap >> prevImgs[Modality::color][1]; cv::cvtColor(prevImgs[Modality::color][1], prevImgs[Modality::grayscale][1], COLOR_BGR2GRAY);
        cap >> prevImgs[Modality::color][0]; cv::cvtColor(prevImgs[Modality::color][0], prevImgs[Modality::grayscale][0], COLOR_BGR2GRAY);
        cap >> currentImg[Modality::color]; cv::cvtColor(currentImg[Modality::color], currentImg[Modality::grayscale], COLOR_BGR2GRAY);
        cap >> nextImgs[Modality::color][0]; cv::cvtColor(nextImgs[Modality::color][0], nextImgs[Modality::grayscale][0], COLOR_BGR2GRAY);
        cap >> nextImgs[Modality::color][1]; cv::cvtColor(nextImgs[Modality::color][1], nextImgs[Modality::grayscale][1], COLOR_BGR2GRAY);
        cap >> nextImgs[Modality::color][2]; cv::cvtColor(nextImgs[Modality::color][2], nextImgs[Modality::grayscale][2], COLOR_BGR2GRAY);

        deque<Mat> binaryFieldImages, candidatePixelImages, magnitudeImages;
        
        // Process the needed number of frames in order to allow the computation of the spatio-temporal correlation
        for (auto i = 0; i < rainParams.numCorrelationFrames; ++i) {
            if (i != 0) {
                int retVal = fetchNextImage(cap, prevImgs, currentImg, nextImgs);

                if (retVal != 0) {
                    cout << "Could not retrive frame " << i << ". Aborting." << endl;
                    return 1;
                }
            }

            Mat candidatePixels; Mat binaryField;

            findCandidatePixels(prevImgs[Modality::grayscale][0], currentImg[Modality::grayscale], nextImgs[Modality::grayscale][0], candidatePixels);
            candidatePixelImages.push_back(candidatePixels);

            enforcePhotometricConstraint(candidatePixels, prevImgs[Modality::grayscale][0], currentImg[Modality::grayscale], binaryField);
            waitKey(0);
            
            binaryFieldImages.push_back(binaryField);
        }

        // Process the first five frames outside the great for-loop to populate the next/prev rain
        // images    
        for (auto i = 0; i < (rainParams.numFramesReplacement * 2 + 1); ++i) {
            Mat candidatePixels, currentMagnitudeImg;

            computeSTCorrelation(binaryFieldImages, currentMagnitudeImg, GNOptions::STZerothTimeLag);

            magnitudeImages.push_back(currentMagnitudeImg);

            for (int method = GNMethod::fullMethod; method <= GNMethod::correlationMagnitude; ++method) {
                for (auto&& param : rainImgs[method]) {
                    param.prevRainImgs[1] = param.prevRainImgs[0].clone();
                    param.prevRainImgs[0] = param.currentRainImg.clone();
                    param.currentRainImg = param.nextRainImgs[0].clone();
                    param.nextRainImgs[0] = param.nextRainImgs[1].clone();
                }
            }

            rainImgs[GNMethod::candidatePixels][0].nextRainImgs[1] = candidatePixelImages.back();
            rainImgs[GNMethod::photometricConstraint][0].nextRainImgs[1] = binaryFieldImages.back();
            rainImgs[GNMethod::correlationMagnitude][0].nextRainImgs[1] = magnitudeImages.back();     

            for (auto&& param : rainImgs[GNMethod::fullMethod]) {
                computeCorrelationDirection(currentMagnitudeImg, param.nextRainImgs[1], 
                    param.customParams.minDirectionalCorrelation[0], param.customParams.maxDirectionalSpread[0]);
            }

            // Fetch a new image, and start over
            int retVal = fetchNextImage(cap, prevImgs, currentImg, nextImgs);

            if (retVal != 0) {
                cout << "Could not retrive frame " << i + rainParams.numCorrelationFrames << ". Aborting." << endl;
                return 1;
            }

            Mat binaryField;
            findCandidatePixels(prevImgs[Modality::grayscale][0], currentImg[Modality::grayscale], nextImgs[Modality::grayscale][0], candidatePixels);
            candidatePixelImages.push_back(candidatePixels);
            candidatePixelImages.pop_front();

            enforcePhotometricConstraint(candidatePixels, prevImgs[Modality::grayscale][0], currentImg[Modality::grayscale], binaryField);
            binaryFieldImages.push_back(binaryField);
            binaryFieldImages.pop_front();
        }

        int numImages = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));

        // We now have enough processed images to remove the rain from the frames

		for (auto i = 0; i < (numImages - rainParams.numCorrelationFrames - (rainParams.numFramesReplacement * 2 + 1)); ++i) {
			int displayedFrameNumber = i + (rainParams.numFramesReplacement + 1) * 2 + rainParams.numCorrelationFrames;
			stringstream outFrameNumber;
			outFrameNumber << setw(5) << setfill('0') << displayedFrameNumber;
			Mat candidatePixels, currentMagnitudeImg;
			cout << "Processing frame " << outFrameNumber.str() << endl;

			if (!rainParams.noGNProcessing)
			{
				computeSTCorrelation(binaryFieldImages, currentMagnitudeImg, GNOptions::STZerothTimeLag);

				magnitudeImages.push_back(currentMagnitudeImg);
				magnitudeImages.pop_front();

				for (int method = GNMethod::fullMethod; method <= GNMethod::correlationMagnitude; ++method) {
					for (auto&& param : rainImgs[method]) {
						param.prevRainImgs[1] = param.prevRainImgs[0].clone();
						param.prevRainImgs[0] = param.currentRainImg.clone();
						param.currentRainImg = param.nextRainImgs[0].clone();
						param.nextRainImgs[0] = param.nextRainImgs[1].clone();
					}
				}

				rainImgs[GNMethod::candidatePixels][0].nextRainImgs[1] = candidatePixelImages.back();
				rainImgs[GNMethod::photometricConstraint][0].nextRainImgs[1] = binaryFieldImages.back();
				rainImgs[GNMethod::correlationMagnitude][0].nextRainImgs[1] = magnitudeImages.back();

				for (auto&& param : rainImgs[GNMethod::fullMethod]) {
					computeCorrelationDirection(currentMagnitudeImg, param.nextRainImgs[1],
						param.customParams.minDirectionalCorrelation[0], param.customParams.maxDirectionalSpread[0]);
				}

				// Remove detected rain. In order to remove the rain from the image, we require 
				// that rain has been detected in: frame-2, frame-1, frame, frame+1, frame+2, 
				// or more/less controlled by the numFramesReplacement parameter.
				// If numFramesReplacement != 2 however, the code below must be changed to 
				// copy the vectors accordingly
				// 
				// The mapping between the images and the rain images is the following:
				// currentImage: nextRainImgs[1]
				// prevImgs[0]: nextRainImgs[0]
				// prevImgs[1]: currentRainImg
				// prevImgs[2]: prevRainImgs[0]or
				// prevImgs[3]: prevRainImgs[1]
				// We must remap the images accordingly in order to remove the rain

				std::vector<Mat> tmpPrevImgs, tmpNextImgs;
				tmpPrevImgs.push_back(prevImgs[Modality::color][2]);
				tmpPrevImgs.push_back(prevImgs[Modality::color][3]);
				tmpNextImgs.push_back(prevImgs[Modality::color][0]);
				tmpNextImgs.push_back(prevImgs[Modality::color][1]);
				map<int, Mat> rainRemovedImg;

				// Remove detected rain by using the full method, candidatePixels, magnitudeImages, and photometric constraint
				for (int method = GNMethod::fullMethod; method <= GNMethod::correlationMagnitude; ++method) {
					for (auto&& param : rainImgs[method]) {
						removeDetectedRain(tmpPrevImgs, prevImgs[Modality::color][1], tmpNextImgs, param, rainRemovedImg[method]);
						cv::imwrite(param.outputFolder + outFrameNumber.str() + ".png", rainRemovedImg[method]);

						if (rainParams.saveDiffImg && (method != GNMethod::fullMethod)) {
							// Compute difference image of intermediate output
							Mat diff;
							absdiff(rainRemovedImg[GNMethod::fullMethod], rainRemovedImg[method], diff);
							cv::imwrite(param.outputFolder + "diff-" + outFrameNumber.str() + ".png", diff * 255);
						}
					}
				}

				if (rainParams.saveOverviewImg) {
					Mat combinedImg = Mat(Size(currentImg[Modality::grayscale].cols * 3, currentImg[Modality::grayscale].rows * 2), currentImg[Modality::grayscale].type()); // CV_8UC1
					Mat upperLeft(combinedImg, Rect(0, 0, currentImg[Modality::grayscale].cols, currentImg[Modality::grayscale].rows));
					prevImgs[Modality::grayscale][1].copyTo(upperLeft);

					Mat upperMid(combinedImg, Rect(currentImg[Modality::grayscale].cols, 0, currentImg[Modality::grayscale].cols, currentImg[Modality::grayscale].rows));
					Mat rainRemovedImgGray;
					cv::cvtColor(rainRemovedImg[GNMethod::fullMethod], rainRemovedImgGray, CV_BGR2GRAY);
					rainRemovedImgGray.copyTo(upperMid);

					Mat upperRight(combinedImg, Rect(currentImg[Modality::grayscale].cols * 2, 0, currentImg[Modality::grayscale].cols, currentImg[Modality::grayscale].rows));
					rainImgs[GNMethod::fullMethod][0].currentRainImg.convertTo(upperRight, CV_8UC1, 100);

					Mat lowerLeft(combinedImg, Rect(0, currentImg[Modality::grayscale].rows, currentImg[Modality::grayscale].cols, currentImg[Modality::grayscale].rows));
					candidatePixelImages[candidatePixelImages.size() - 3].copyTo(lowerLeft);

					Mat lowerMid(combinedImg, Rect(currentImg[Modality::grayscale].cols, currentImg[Modality::grayscale].rows, currentImg[Modality::grayscale].cols, currentImg[Modality::grayscale].rows));
					binaryFieldImages[binaryFieldImages.size() - 3].copyTo(lowerMid);

					Mat lowerRight(combinedImg, Rect(currentImg[Modality::grayscale].cols * 2, currentImg[Modality::grayscale].rows, currentImg[Modality::grayscale].cols, currentImg[Modality::grayscale].rows));
					magnitudeImages[magnitudeImages.size() - 3].convertTo(lowerRight, CV_8UC1, 100);
					//prevMagnitudeImgs[1].convertTo(lowerRight, CV_8UC1, 100);
					cv::imwrite(rainImgs[GNMethod::overview][0].outputFolder + outFrameNumber.str() + ".png", combinedImg);
				}
			}

            if (rainParams.useMedianBlur) {
                Mat medBlur;
                medianBlur(prevImgs[Modality::color][1], medBlur, 3);
                cv::imwrite(outputFolder + "/Median/" + outFrameNumber.str() + ".png", medBlur);
            }

            // Grab a new frame, and start the processing over
            int retVal = fetchNextImage(cap, prevImgs, currentImg, nextImgs);

            if (retVal != 0) {
                cout << "Could not retrive frame " << i + rainParams.numCorrelationFrames + rainParams.numFramesReplacement * 2 + 1 << ". Aborting." << endl;
                return 0;
            }

			if (!rainParams.noGNProcessing)
			{
				Mat binaryField;

				findCandidatePixels(prevImgs[Modality::grayscale][0], currentImg[Modality::grayscale], nextImgs[Modality::grayscale][0], candidatePixels);
				candidatePixelImages.push_back(candidatePixels);
				candidatePixelImages.pop_front();

				//imwrite("Out/CandidatePixels-" + std::to_string(i + 1 + rainParams.numCorrelationFrames) + ".png", candidatePixels);

				enforcePhotometricConstraint(candidatePixels, prevImgs[Modality::grayscale][0], currentImg[Modality::grayscale], binaryField);
				//imwrite("Out/PConstraint-" + std::to_string(i + 1 + rainParams.numCorrelationFrames) + ".png", binaryField);

				binaryFieldImages.push_back(binaryField);
				binaryFieldImages.pop_front();
			}
        }

    }
    else {
        cout << "Could not open video " << inputVideo << endl; 
        return 1;
    }

    return 0;
}

//! Find candidate rain pixels from the grayscale input images
/*!
\param prevImg previous image, frame n - 1 
\param currentImg current image, frame n
\param nextImg next image, frame n + 1
\param outImg returned image with the found candidate rain pixels
*/
void GargNayarRainRemover::findCandidatePixels(cv::Mat prevImg, cv::Mat currentImg, cv::Mat nextImg, cv::Mat &outImg)
{
    assert(prevImg.size() == currentImg.size());
    assert(nextImg.size() == currentImg.size());

    outImg = Mat::zeros(currentImg.size(), CV_8UC1);

    Mat prevDiff, nextDiff;
    subtract(currentImg, prevImg, prevDiff);
    subtract(currentImg, nextImg, nextDiff);

    // If the previous difference equals then next difference at pixel level, 
    // and the pixel value in prevDiff/nextDiff is larger than a parameter, 
    // mark this as a rain candidate
    Mat diffEquality, invDiffEquality;
    absdiff(prevDiff, nextDiff, diffEquality); // if prevDiff == nextDiff, 0
    threshold(diffEquality, invDiffEquality, 0, 255, THRESH_BINARY_INV); // if prevDiff == nextDiff, 255

    // Only keep the values in prevDiff that equals nextDiff
    Mat maskedDiff;
    prevDiff.copyTo(maskedDiff, invDiffEquality);

    // Check prevDiff >= rainParams.c.
    // If this applies, the intensity change just occurs for this frame and is 
    // bright enough to be detected as a rain chandidate.  
    // Mark this in the out image, outImg.at<uchar>(y, x) = 255
    inRange(maskedDiff, rainParams.c, 255, outImg);
}

//! Enfore the linear photometric constraint from Garg and Nayar
/*!
\param prevImg previous image, frame n - 1
\param currentImg current image, frame n
\param nextImg next image, frame n + 1
\param outImg returned image with the extracted rain streaks
*/
void GargNayarRainRemover::enforcePhotometricConstraint(cv::Mat inBinaryImg, cv::Mat prevImg, cv::Mat currentImg, cv::Mat &outImg)
{
    // Enforce the linear photometric constraint introduced by Garg and Nayar, 2004

    // Step 1: Find the connected components (individual rain streaks) of the binary image
    Mat inputImage = inBinaryImg.clone();
    vector<vector<Point> > contours;

    findContours(inputImage, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    // Step 2: Filter the connected components by calculating the fit of the pixels 
    // with first-order a linear model
    outImg = inBinaryImg.clone();
    int deletedContours = 0;
    
    for (auto i = 0; i < contours.size(); ++i) {
        if (contours[i].size() > 1) {
            // If the contour is larger than 1 pixel, we check for linearity
            // In this case, we solve the (possibly overdetermined) system of linear equations
            // for each pixel in the contour:
            //   deltaI = -beta * prevImg + alpha
            //   where:
            //     deltaI = currentImg - prevImg
            //     beta, alpha = parameters that should be estimated
            //
            // We can write the linear equation in a matrix form:
            //              ( beta  )
            // (-prevImg 1) ( alpha ) = deltaI
            // 
            // or: 
            //   ( beta  )
            // A ( alpha ) = deltaI
            // 
            // where:
            //    A      = (-prevImg 1), a j,2 matrix
            //    deltaI : a j,1 matrix
            //
            // The equation is solved by least squares by:
            // ( beta )
            // ( alpha ) = (A^T A)^(-1) A^T deltaI
            
            Mat A = Mat::ones(Size(2, static_cast<int>(contours[i].size())), CV_32FC1);
            Mat deltaI = Mat::ones(Size(1, static_cast<int>(contours[i].size())), CV_32FC1);


            // Step 3: Populate the matrices A and deltaI with the values from the images
            for (auto j = 0; j < contours[i].size(); ++j) {
                A.at<float>(j, 0) = -static_cast<float>(prevImg.at<uchar>(contours[i][j].y, contours[i][j].x));

                int prevDiff = currentImg.at<uchar>(contours[i][j].y, contours[i][j].x) -
                    prevImg.at<uchar>(contours[i][j].y, contours[i][j].x);
                deltaI.at<float>(j, 0) = static_cast<float>(prevDiff);
            }

            // Step 4: Find the best fit

            // Compute(A^T A)^(-1) A^T deltaI
            Mat estimate = (A.t() * A).inv() * A.t() * deltaI;
            float beta = estimate.at<float>(0, 0);

            // If the estimated beta is above the threshold, we should discard the current streak.
            // Otherwise, keep the streak
            if (estimate.at<float>(0, 0) > rainParams.betaMax) {
                // Make sure the entire contour is deleted from the output image
                for (auto j = 0; j < contours[i].size(); ++j) {
                    outImg.at<uchar>(contours[i][j].y, contours[i][j].x) = 0;
                }

                ++deletedContours;
            }
        }

        // If the contour just consists of one point, there is no point in checking, and we might thus
        // let the contour pass the test. Because we have copied the binary input image, we need not to do anything        
    }

    if (rainParams.verbose) {
        cout << "PMConstraint: Found " << contours.size() << " contours in rain candidate image, and deleted " << deletedContours << " of those " << endl;
    }
}

//! Compute the spatio-temporal correlation of a collection of rain streak images
/*!
\param binaryFieldHistory extracted rain streaks under the photometric constraint from the previous n frames
\param outImg filtered rain image (binary) subject to the spatio-temporal correlation of the provided history
\param method Time lag method. Use GNOptions:STZerothTimeLag to reproduce the original results by Garg and Nayar
*/
void GargNayarRainRemover::computeSTCorrelation(std::deque<cv::Mat> binaryFieldHistory, cv::Mat &outImg, int method)
{
    // Compute the spatio-temporal correlation of the binary field history
    assert(static_cast<int>(binaryFieldHistory.size()) == rainParams.numCorrelationFrames);

    if (rainParams.numCorrelationFrames <= 0) {
        return;
    }

    outImg = Mat::zeros(binaryFieldHistory.front().size(), CV_32FC1);

    // Compute the spatial correlation for each binary field, and add them to 
    // the temporary spatio-temporal image
    for (auto it = binaryFieldHistory.cbegin(); it != binaryFieldHistory.cend(); ++it) {
        Mat bField;

        if (method == GNOptions::STZerothTimeLag) {
            // Add the spatial correlation to the output, spatio-temporal image to produce the 
            // zero'th order temporal correlation as presented in the original paper
            bField = *it;
        }
        else if (method == GNOptions::STVaryingTimeLag) {
            // Use an alternative interpretation of the the spatio-temporal correspondence 
            // what uses a varying time lag - thus, use the binary field of the 'current' image
            // to multiply in the end of the for loop
            bField = binaryFieldHistory.back();
        }
        

        // Create a scaled version of the binary field where the only values are 0,1
        Mat sBField;
        threshold(bField, sBField, 254, 1, THRESH_BINARY); // If val > threshold, then 1, else 0

        // Compute the spatial correlation for all pixels of the current image using a box filter
        // The correlation matrix is used subsequently to compute the correlation Rb
        Mat filteredImg = Mat::zeros(bField.size(), bField.type());
        int window = rainParams.neighborhoodSize;

        boxFilter(sBField, filteredImg, -1, Size(window, window), Point(-1, -1), false);

        // Perform a per-element multiplication of the scaled binary field and the 
        // correlation matrix to compute Rb for the particular instance of the binary field
        Mat spatialCorr;
        multiply(sBField, filteredImg, spatialCorr);
        
        add(spatialCorr, outImg, outImg, noArray(), outImg.depth());
    }   

    // Average output image with a 3x3 mean filter
    Mat nonBlurImg = outImg.clone();
    blur(nonBlurImg, outImg, Size(3, 3), Point(-1, -1), BORDER_REPLICATE);
    
    // We do not normalize the output by the term 1/L as presented in equation 3 in (Garg, Nayar).
    // However, this only affects the magnitude of the spatio-temporal correlation linearly, which 
    // is compensated by adjusting the rainParams.minDirectionalCorrelation accordingly
}


//! Compute the directional correlation of the spatio-temporal correlation. This will exclude non-streak-like rain streaks
/*!
Prior to calling this method, the directional masks should have been generated by calling computeDirectionalMasks()

\param ingImg binary image as output by computeSTCorrelation
\param outImg filtered rain image
\param minDirectionalCorrelation a rain pixel should have a directional correlation over this threshold in order to remain a rain streak
\param maxDirectionalSpread the directional correlation of a rain pixel should not vary more than this threshold
*/
void GargNayarRainRemover::computeCorrelationDirection(cv::Mat inImg, cv::Mat &outImg, 
    float minDirectionalCorrelation, int maxDirectionalSpread)
{
    // Find the correlation of the spatio-temporal image with the directional masks
    // Use template mathing for each pixel/area in the input image
    vector<Mat> matchResults;

    for (auto i = 0; i < directionalMasks.size(); ++i) {
        Mat result;
        matchTemplate(inImg, directionalMasks[i], result, TM_CCORR_NORMED);
        matchResults.push_back(result);
        //cv::imwrite("matchRes-" + std::to_string(i) + ".png", result*100);
    }

    // Go through every pixel in the matched images and find areas where:
    // a) The correlation is weak (defined by arbitrary parameter)
    // b) There is consistency in the directional correlation

    outImg = inImg.clone();

    int discardedRainPixels = 0;

    int templateBoundary = (rainParams.neighborhoodSize - 1) / 2;

    for (auto x = templateBoundary; x < inImg.cols-templateBoundary; ++x) {
        for (auto y = templateBoundary; y < inImg.rows-templateBoundary; ++y) {

            if (inImg.at<float>(y, x) != 0) {
                // Only proceed for candidate rain pixels

                map<int, float> correlations; float maxCorrelation = 0;
                int maxCorrelationIndex = -1;

                int matchX = x - templateBoundary;
                int matchY = y - templateBoundary;

                for (auto i = 0; i < directionalMasks.size(); ++i) {
                    if (matchResults[i].at<float>(matchY, matchX) > minDirectionalCorrelation) {
                        correlations[i] = matchResults[i].at<float>(matchY, matchX);

                        if (matchResults[i].at<float>(matchY, matchX) > maxCorrelation) {
                            maxCorrelation = matchResults[i].at<float>(matchY, matchX);
                            maxCorrelationIndex = i;
                        }
                    }
                }

                // From the correlations, figure out if this pixel does not correlate to a rain pixel
                bool rainPixel = true;

                if (maxCorrelation == 0) {
                    // No correlation was above the minimum criteria. Delete this pixel
                    rainPixel = false;
                }
                else {
                    // The correlation was above the criteria. Check if there is consistency 
                    // in the directions. This means, that any other correlation in the 
                    // correlations map must be within (param)*10 degrees of the largest 
                    // correlation value
                    // Go through the correlation maps and check for this
                    for (auto const &ent1 : correlations) {
                        if (std::abs(ent1.first - maxCorrelationIndex) > maxDirectionalSpread) {
                            rainPixel = false;
                            break;
                        }
                    }
                }

                if (!rainPixel)     
                {
                    // The current pixel is disqualified as a rain pixel. Delete it from the output map
                    outImg.at<float>(y, x) = 0;
                    ++discardedRainPixels;
                }

                // If the current pixel is qualified as a rain pixel (hurray!), we don't need to do anything, 
                // as we have cloned the input image
            }
        }
    }

    if (rainParams.verbose) {
        cout << "Discarded " << discardedRainPixels << " pixels from ST rain image" << endl;
    }

    
}

//! Compute the directional masks. 
void GargNayarRainRemover::computeDirectionalMasks()
{
    // Construct 18 (the paper says 17, but the range from {0, 10, ... 170} indicates that we need 18)
    // oriented binary masks that helps compute the correlation direction
    
    Mat horizontalMask = Mat::zeros(Size(rainParams.neighborhoodSize, rainParams.neighborhoodSize), CV_32FC1);
    cv::line(horizontalMask, Point(0, (rainParams.neighborhoodSize - 1) / 2), Point(rainParams.neighborhoodSize, (rainParams.neighborhoodSize - 1) / 2), Scalar(255));
    directionalMasks.push_back(horizontalMask);

    // Rotate the horizontal mask by steps of 10 degrees
    for (auto i = 1; i < 18; ++i) {
        Mat rot = getRotationMatrix2D(Point((rainParams.neighborhoodSize - 1) / 2, (rainParams.neighborhoodSize - 1) / 2), i * 10, 1);
        Mat rotMask;

        warpAffine(horizontalMask, rotMask, rot, horizontalMask.size());
        directionalMasks.push_back(rotMask);
    }
}

//! Remove rain from the current image if rain is detected using the current rain image. 
//! If there is rain detected at a pixel in the current rain image, look at the previous and next rain image. 
//! If there is no rain in these, use the averaged values
//! of these images to replace the rain-affected pixel.
void GargNayarRainRemover::removeDetectedRain(std::vector<cv::Mat> prevImgs, cv::Mat currentImg, std::vector<cv::Mat> nextImgs, 
    GNRainImage rainImage, cv::Mat &rainRemovedCurrentImg)
{


    assert(prevImgs.size() == nextImgs.size() && rainImage.prevRainImgs.size() == rainImage.nextRainImgs.size());

    for (auto i = 0; i < prevImgs.size(); ++i) {
        assert(prevImgs[i].size().height == currentImg.size().height && nextImgs[i].size().height == rainImage.currentRainImg.size().height);
        assert(prevImgs[i].size().width == currentImg.size().width && nextImgs[i].size().width == rainImage.currentRainImg.size().width);
        assert(rainImage.prevRainImgs[i].size().height == rainImage.currentRainImg.size().height && rainImage.currentRainImg.size().height == rainImage.nextRainImgs[i].size().height);
        assert(rainImage.prevRainImgs[i].size().width == rainImage.currentRainImg.size().width && rainImage.currentRainImg.size().width == rainImage.nextRainImgs[i].size().width);
    }

    currentImg.copyTo(rainRemovedCurrentImg);

    // Beware; Duplicate code due to different type specifiers!

    if (rainImage.currentRainImg.type() == CV_8UC1) {
        for (auto x = 0; x < currentImg.cols; ++x) {
            for (auto y = 0; y < currentImg.rows; ++y) {
                if (rainImage.currentRainImg.at<uchar>(y, x) != 0) {
                    // Rain has been detected in the current pixel
                    // Find a previous + next pixel to replace it

                    // Make sensible defaults if all previous/next pixels are affected by rain 
                    Vec3b replacementPrev = prevImgs[0].at<Vec3b>(y, x);
                    Vec3b replacementNext = nextImgs[0].at<Vec3b>(y, x);

                    for (auto i = 0; i < rainImage.prevRainImgs.size(); ++i) {
                        if (rainImage.prevRainImgs[i].at<uchar>(y, x) == 0) {
                            // No rain detected. Use this pixel
                            replacementPrev = prevImgs[i].at<Vec3b>(y, x);
                            break;
                        }
                    }

                    for (auto i = 0; i < rainImage.nextRainImgs.size(); ++i) {
                        if (rainImage.nextRainImgs[i].at<uchar>(y, x) == 0) {
                            // No rain detected. Use this pixel
                            replacementNext = nextImgs[i].at<Vec3b>(y, x);
                            break;
                        }
                    }

                    // Average the proposed previous and next pixels and replace the current pixel
                    rainRemovedCurrentImg.at<Vec3b>(y, x)[0] = (replacementPrev[0] + replacementNext[0]) / 2;
                    rainRemovedCurrentImg.at<Vec3b>(y, x)[1] = (replacementPrev[1] + replacementNext[1]) / 2;
                    rainRemovedCurrentImg.at<Vec3b>(y, x)[2] = (replacementPrev[2] + replacementNext[2]) / 2;
                }
            }
        }
    }
    else if (rainImage.currentRainImg.type() == CV_32FC1) {
        for (auto x = 0; x < currentImg.cols; ++x) {
            for (auto y = 0; y < currentImg.rows; ++y) {
                if (rainImage.currentRainImg.at<float>(y, x) != 0) {
                    // Rain has been detected in the current pixel
                    // Find a previous + next pixel to replace it

                    // Make sensible defaults if all previous/next pixels are affected by rain
                    Vec3b replacementPrev = prevImgs[0].at<Vec3b>(y, x);
                    Vec3b replacementNext = nextImgs[0].at<Vec3b>(y, x);

                    for (auto i = 0; i < rainImage.prevRainImgs.size(); ++i) {
                        if (rainImage.prevRainImgs[i].at<float>(y, x) == 0) {
                            // No rain detected. Use this pixel
                            replacementPrev = prevImgs[i].at<Vec3b>(y, x);
                            break;
                        }
                    }

                    for (auto i = 0; i < rainImage.nextRainImgs.size(); ++i) {
                        if (rainImage.nextRainImgs[i].at<float>(y, x) == 0) {
                            // No rain detected. Use this pixel
                            replacementNext = nextImgs[i].at<Vec3b>(y, x);
                            break;
                        }
                    }

                    // Average the proposed previous and next pixels and replace the current pixel
                    rainRemovedCurrentImg.at<Vec3b>(y, x)[0] = (replacementPrev[0] + replacementNext[0]) / 2;
                    rainRemovedCurrentImg.at<Vec3b>(y, x)[1] = (replacementPrev[1] + replacementNext[1]) / 2;
                    rainRemovedCurrentImg.at<Vec3b>(y, x)[2] = (replacementPrev[2] + replacementNext[2]) / 2;
                }
            }
        }
    }

}


int main(int argc, char** argv)
{
    const string keys =
        "{help h            |       | Print help message}"
        "{fileName fn       |       | Input video file}"
        "{outputFolder of   |       | Output folder of processed files}"
        "{settingsFile      |       | File from which settings are loaded/saved}"
        "{saveSettings s    | 0     | Save settings to settingsFile (0,1)}"
        "{overviewImg oi    | 0     | Generate overview images of deraining process (0,1)}"
        "{diffImg di        | 0     | Generate difference frames from intermediate steps of algorithm (0,1)}"
        "{medianBlur mb     | 0     | Generate derained frames using basic mean blur (3x3 kernel) (0,1)}"
		"{noGNProcessing    | 0     | Disables the entire Garg-Nayar processing and only allows median blur (0,1)}"
        "{verbose v         | 0     | Write additional debug information to console (0,1)}"
        ;

    cv::CommandLineParser cmd(argc, argv, keys);
	cmd.about("Reimplementation of the method from Garg and Nayar in \"Detection and Removal of Rain From Videos\".");

    if (argc <= 1 || (cmd.has("help"))) {
        std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
        std::cout << "Available options:" << std::endl;
        cmd.printMessage();
        return 1;
    }

    std::string filename = cmd.get<std::string>("fileName");
    std::string outputFolder = cmd.get<std::string>("outputFolder");

    GNRainParameters defaultParams = GargNayarRainRemover::getDefaultParameters();

    // Load settings if settingsFile exists
    if (cmd.has("settingsFile")) {
        defaultParams = GargNayarRainRemover::loadParameters(cmd.get<std::string>("settingsFile"));
    }

    defaultParams.saveOverviewImg = cmd.get<int>("overviewImg") != 0;
    defaultParams.useMedianBlur = cmd.get<int>("medianBlur") != 0;
    defaultParams.saveDiffImg = cmd.get<int>("diffImg") != 0;
    defaultParams.verbose = cmd.get<int>("verbose") != 0;
	defaultParams.noGNProcessing = cmd.get<int>("noGNProcessing") != 0;
	

    GargNayarRainRemover gNRainRemover(filename, outputFolder, defaultParams);

    // Save final settings
    if (cmd.has("settingsFile") && (cmd.get<int>("saveSettings") != 0)) {
        gNRainRemover.saveParameters(cmd.get<std::string>("settingsFile"));
    }

    gNRainRemover.removeRain();    

    //std::string filename = "C:/Code/Video/RainSnow/OstreAlle-DagHammerskjoldsGade/20130613-06-10-00-cam1.mkv";
    //std::string outputFolder = "D:/RainSnow/RainRemoval/OstreAlle-DagHammerskjoldsGade/RainRemoval/0613-06-10-00/GargNayar/";

    


}