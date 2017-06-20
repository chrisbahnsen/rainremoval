// MIT License
// 
// Copyright(c) 2017 Aalborg University
// Chris Holmberg Bahnsen, June 2017
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


// An open source implementation of the estimation of rain intensity algorithm as 
// presented by J. Bossu, N. Hautière, and J.P. Tarel in "Rain or snow detection 
// in image sequences through use of a histogram of orientation of streaks." 
// appearing in International Journal of Computer Vision, 2011
//
// See BossuRainGauge.h for further explanation



#include "BossuRainGauge.h"


int main()
{
    return 0;
}

BossuRainIntensityMeasurer::BossuRainIntensityMeasurer(std::string inputVideo, std::string outputFolder, BossuRainParameters rainParams)
{
	this->inputVideo = inputVideo;
	this->outputFolder = outputFolder;
	this->rainParams = rainParams;
}

int BossuRainIntensityMeasurer::detectRain()
{
	// Open video
	VideoCapture cap(inputVideo);

	cv::Ptr<BackgroundSubtractorMOG2> backgroundSubtractor = 
		cv::createBackgroundSubtractorMOG2(500, 16.0, false);

	if (cap.isOpened()) {
		Mat frame, foregroundMask, backgroundImage, foregroundImage;

		// Use the first six frames to initialize the Gaussian Mixture Model
		for (auto i = 0; i < 7; ++i) {
			cap.read(frame);

			backgroundSubtractor->apply(frame, foregroundMask);
		}

		while (cap.grab()) {
			// Continue while there are frames to retrieve
			
			backgroundSubtractor->apply(frame, foregroundMask);
			backgroundSubtractor->getBackgroundImage(backgroundImage);

			// Construct the foreground image from the mask
			frame.copyTo(foregroundImage, foregroundMask);

			// Use the Garg-Nayar intensity constraint to select candidate rain pixels
			Mat diffImg = foregroundImage - backgroundImage;
			Mat candidateRainMask;

			threshold(diffImg, candidateRainMask, rainParams.c, 255, CV_THRESH_BINARY);

			// We now have the candidate rain pixels. Use connected component analysis
			// to filter out large connected components
			vector<vector<Point> > contours, filteredContours; 

			findContours(candidateRainMask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

			Mat filteredCandidateMask = candidateRainMask.clone();

			for (auto contour : contours) {
				if (contour.size() > rainParams.maximumBlobSize) {
					// Delete the contour from the rain image
					if (rainParams.verbose) {
						std::cout << "Deleting contour of size " << contour.size();
					}

					for (auto point : contour) {
						filteredCandidateMask.at<uchar>(point.y, point.x) = 0;
					}
				}
				else {
					// Retain the contour if below or equal to the size threshold
					filteredContours.push_back(contour);
				}
			}

			// Compute the Histogram of Orientation of Streaks (HOS) from the contours


		}

	}

	return 0;
}

BossuRainParameters BossuRainIntensityMeasurer::getDefaultParameters()
{
	return BossuRainParameters();
}

void BossuRainIntensityMeasurer::computeOrientationHistogram(const std::vector<std::vector<cv::Point>>& contours, std::vector<double>& histogram)
{
	// Compute the moments from the contours and get the orientation of the BLOB

	for (auto contour : contours) {
		Moments mu = moments(contour, false);

		// Compute the major semiaxis of the ellipse equivalent to the BLOB
		// In order to do so, we must compute eigenvalues of the matrix
		// | m20 m11 |
		// | m11 m02 |
		// Bossu et al, 2011, pp. 6, equation 16
		Mat momentsMat = Mat(2, 2, CV_64FC1);
		momentsMat.at<double>(0, 0) = mu.m20;
		momentsMat.at<double>(1, 0) = mu.m11;
		momentsMat.at<double>(0, 1) = mu.m11;
		momentsMat.at<double>(1, 1) = mu.m02;

		Mat eigenvalues;
		eigen(momentsMat, eigenvalues);

		// Extract the largest eigenvalue and compute the major semi-axis according to
		// Bossu et al, 2011, pp. 6, equation 14
		double a = 2 * sqrt(eigenvalues.at<double>(0, 0));

		// Bossu et al, 2011, pp. 6, equation 17
		double orientation = 0.5 * atan2(2 * mu.m11, (mu.m02 - mu.m20));

		if (rainParams.verbose) {
			std::cout << "Orientation: " << orientation << endl;
		}

		// Convert to degrees and scale in range [0, \pi] ( [0 180] )
		orientation = orientation * 180. / CV_PI + 180;

		if (rainParams.verbose) {
			std::cout << "To degrees, converted: " << orientation << endl;
		}

		// Compute the uncertainty of the estimate to be used as standard deviation
		// for Gaussian
		double estimateUncertainty = sqrt((pow(mu.m02 - mu.m20, 2) + 2 * pow(mu.m11, 2)) /
			(pow(mu.m02 - mu.m20, 2) + 4 * pow(mu.m11, 2)));


		// Compute the Gaussian (Parzen) estimate of the true orientation and 
		// add to the histogram
		for (double angle = 0; angle < histogram.size(); ++angle) {
			histogram[angle] += a / (estimateUncertainty * sqrt(180)) *
				exp(-0.5 * pow((angle - orientation) / estimateUncertainty, 2));
		}
	}

	

}

void BossuRainIntensityMeasurer::estimateGaussianUniformMixtureDistribution(const std::vector<double>& histogram, 
	int numberOfObservations,
	double & gaussianMean, 
	double & gaussianStdDev, 
	double & gaussianMixtureProportion)
{
	// Estimate a Gaussian-Uniform mixture distribution using the data in the histogram. 
	// Use the Expectation-Maximization algorithm as provided by Bossu et al, 2011, page 8, eq. 24-25

	// Initialize the EM algorithm as described by Bossu et al, 2011, page 8, upper right column
	// We find the median value of the histogram and use it to estimate initial values of
	// gaussianMean, gaussianStdDev, and gaussianMixtureProportion
	
	// In order to find median, find sum of histogram (the histogram is our unnormalized PDF)
	double histogramSum = 0;

	for (auto& n : histogram) {
		histogramSum += n;
	}

	// Compute the Cumulative Density Function (CDF) of the histogram and stop when we have 
	// reached 50 % of the total sum
	double cumulativeSum = 0;
	double fiftyPercentSum = histogramSum / 2.;
	double median;

	for (auto i = 0; i < histogram.size(); ++i) {
		cumulativeSum += histogram[i];

		if (cumulativeSum > fiftyPercentSum) {
			median = i;
			break;
		}
	}

	// Now that we have found the median, only use entries in the histogram equal to or above
	// the position of the median to calculate the mean, std.dev and proportion estimate
	double sumAboveMedian = 0, observationSum = 0;
	double initialStdDev = 0;
	double initialMixtureProportion = 0;
	double totalSum = 0;


	for (auto i = median; i < histogram.size(); ++i) {
		sumAboveMedian += i * histogram[i];
		observationSum += histogram[i];
	}

	double initialMean = sumAboveMedian / observationSum;

	double sumOfSqDiffToMean = 0;

	for (auto i = median; i < histogram.size(); ++i) {
		sumOfSqDiffToMean += pow(i * histogram[i] - initialMean, 2);
	}

	initialStdDev = sqrt(sumOfSqDiffToMean / observationSum);

}
