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

using namespace std;
using namespace cv;

BossuRainIntensityMeasurer::BossuRainIntensityMeasurer(std::string inputVideo, std::string filePath, std::string settingsFile, std::string outputFolder, BossuRainParameters rainParams)
{
	this->inputVideo = inputVideo;
	this->filePath = filePath;
	this->settingsFile = settingsFile;
	this->outputFolder = outputFolder;
	this->rainParams = rainParams;
}

int BossuRainIntensityMeasurer::detectRain()
{
	// Open video
	VideoCapture cap(this->filePath + this->inputVideo);

	cv::Ptr<BackgroundSubtractorMOG2> backgroundSubtractor = 
		cv::createBackgroundSubtractorMOG2(500, 16.0, false);

	// Create file, write header
	ofstream resultsFile;
	resultsFile.open(this->outputFolder + "/" + this->inputVideo + "_" + "Results" + ".csv", ios::out | ios::trunc);

	std::string header;
	header += string("settingsFile") + ";" + "InputVideo" + "; " + "Frame#" + "; " +
		"GaussMean" + ";" + "GaussStdDev" + ";" +
		"GaussMixProp" + ";" + "Goodness-Of-Fit Value" + ";" +
		"kalmanGaussMean" + ";" + "kalmanGaussStdDev" + ";" +
		"kalmanGaussMixProp" + ";" + "Rain Intensity" + ";" + "Kalman Rain Intensity" + "\n";
	 
	resultsFile << header;

	double kalmanMeanTmp, kalmanStdDevTmp, kalmanMixPropTmp = 0.0; 0.0; 0.0;

	if (cap.isOpened()) {
		Mat frame, foregroundMask, backgroundImage, foregroundImage;

		// Use the first six frames to initialize the Gaussian Mixture Model
		for (auto i = 0; i < 7; ++i) {
			cap.read(frame);

			backgroundSubtractor->apply(frame, foregroundMask);
			resultsFile << "\n"; // Write blank frame
		}

		// Set up Kalman filter to smooth the Gaussian-Uniform mixture distribution
		cv::KalmanFilter KF(6, 3, 0, CV_64F);
		
		Mat state(6, 1, CV_64F); // mu, std.dev, gaussianMixtureDist, delta_mu,
								 // delta_std.dev, delta_gaussianMixtureDist
		KF.transitionMatrix = (Mat_<double>(6, 6) << 
			1, 0, 0, 1, 0, 0,
			0, 1, 0, 0, 1, 0,
			0, 0, 1, 0, 0, 1,
			0, 0, 0, 1, 0, 0,
			0, 0, 0, 0, 1, 0,
			0, 0, 0, 0, 0, 1);
		setIdentity(KF.measurementMatrix);
		setIdentity(KF.processNoiseCov, Scalar::all(0.01)); // According to Bossu et al, 2011, p. 10, right column
		setIdentity(KF.measurementNoiseCov, Scalar::all(0.1)); // According to Bossu et al, 2011, p. 10, right column
		setIdentity(KF.errorCovPost, Scalar::all(1));
		
		int vidLength = cap.get(CV_CAP_PROP_FRAME_COUNT);
		int frameCounter = 6;
		while (cap.grab()) {
			frameCounter++;

			if (frameCounter % 100 == 0)
				cout << "Frame: " << frameCounter << "/" << vidLength << endl;

			// Continue while there are frames to retrieve
			cap.retrieve(frame);
			
			backgroundSubtractor->apply(frame, foregroundMask);
			backgroundSubtractor->getBackgroundImage(backgroundImage);

			// Construct the foreground image from the mask
			frame.copyTo(foregroundImage, foregroundMask);

			// Operate on grayscale images from now on
			Mat grayForegroundImage, grayBackgroundImage;
			cvtColor(foregroundImage, grayForegroundImage, COLOR_BGR2GRAY);
			cvtColor(backgroundImage, grayBackgroundImage, COLOR_BGR2GRAY);

			// Use the Garg-Nayar intensity constraint to select candidate rain pixels
			Mat diffImg = grayForegroundImage - grayBackgroundImage;
			Mat candidateRainMask;

			threshold(diffImg, candidateRainMask, rainParams.c, 255, CV_THRESH_BINARY);

			// We now have the candidate rain pixels. Use connected component analysis
			// to filter out large connected components in candidateRainMask
			vector<vector<Point> > contours, filteredContours; 

			Mat contourRainMask = candidateRainMask.clone();
			findContours(contourRainMask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

			if (rainParams.debug) {
				cout << "Sum of non-zero in candidateRainMask " << cv::countNonZero(candidateRainMask) << endl;
				cout << "Number of contours: " << contours.size() << endl;
			}

			// Copy of candidateRainMask to keep track of already processed pixels
			Mat rainPixelsTracker = candidateRainMask.clone();

			//Matrix to mask out all unusable contours/BLOBs
			Mat mask = Mat::ones(candidateRainMask.rows, candidateRainMask.cols, CV_8UC1) * 255;

			//Sort contours by contour size in ascending order
			sort(contours.begin(), contours.end(), [](const vector<Point>& c1, const vector<Point>& c2) {
				return c1.size() < c2.size();
			});

			// Threshold the detected blob. If outside threshold write to the mask and set to 0
			int deletedContours = 0;
			for (int i = 0; i < contours.size(); i++) {
				// Draw all pixels inside the contour
				Mat contourMask = Mat::zeros(candidateRainMask.rows, candidateRainMask.cols, CV_8UC1);
				cv::drawContours(contourMask, contours, i, cv::Scalar(255, 255, 255), cv::FILLED);

				//Find which pixels are actually inside the drawn contour, and count them
				bitwise_and(contourMask, rainPixelsTracker, contourMask);
				int blobSize = cv::countNonZero(contourMask);

				// Check if a BLOB only consists of contour pixels. If so it is discarded, by setting blob size to 0
				blobSize = (blobSize - contours[i].size()) > 0 ? blobSize : 0;

				if ((blobSize > rainParams.maximumBlobSize) ||
					(blobSize < rainParams.minimumBlobSize)) {
					mask = mask - contourMask;
					deletedContours++;
				}
				else {
					filteredContours.push_back(contours[i]);
				}
				//Remove the BLOB pixels so that they arent counted again by other BLOBS by accident, as long as they werent just an empty blob)
				if (blobSize > 0)
					rainPixelsTracker -= contourMask;
			}

			// Apply bitwise and on the candidateRainMask wth the Mask matrix
			Mat filteredCandidateMask = candidateRainMask.clone();
			bitwise_and(filteredCandidateMask, mask, filteredCandidateMask);

			// Delete the contour from the rain image
			if (rainParams.debug) {
				std::cout << "Deleted " << deletedContours << " contours with dm: " << rainParams.dm << endl;
			}


			if (rainParams.saveImg) {
				imwrite("Image.png", frame);
				imwrite("backgroundImg.png", backgroundImage);
				imwrite("foregroundImage.png", grayForegroundImage);
				imwrite("diffImg.png", diffImg);
				imwrite("Candidate.png", candidateRainMask);
				imwrite("Filtered.png", filteredCandidateMask);
				imwrite("Mask.png", mask);
			}
			if (rainParams.debug) {
				imshow("Image", frame);
				imshow("backgroundImg", backgroundImage);
				imshow("foregroundImage", grayForegroundImage);
				imshow("diffImg", diffImg);
				imshow("Candidate", candidateRainMask);
				imshow("Filtered", filteredCandidateMask);
				imshow("Mask", mask);
				waitKey(0);
			}

			// 4. Compute the Histogram of Orientation of Streaks (HOS) from the contours
			vector<double> histogram;
			computeOrientationHistogram(filteredContours, histogram, rainParams.dm);

			// 5. Model the accumulated histogram using a mixture distribution of 1 Gaussian
			//    and 1 uniform distribution in the range [0-179] degrees.
			double gaussianMean, gaussianStdDev, gaussianMixtureProportion;

			estimateGaussianUniformMixtureDistribution(histogram, filteredContours.size(),
				gaussianMean, gaussianStdDev, gaussianMixtureProportion);

			// 6. Goodness-Of-Fit test between the observed histogram and estimated normal distribution
			double ksTest = goodnessOfFitTest(histogram, gaussianMean, gaussianStdDev, gaussianMixtureProportion);

			if(rainParams.debug || rainParams.saveImg)
				plotGoodnessOfFitTest(histogram, gaussianMean, gaussianStdDev, gaussianMixtureProportion);

			// 7. Use a Kalman filter for each of the three parameters of the mixture
			//	  distribution to smooth the model temporally
			//    (Should only update if the Goodness-OF-Fit test is within the defiend threshold, but we still enter here for plotting reasons. Though we only update if ksTest is met)
				
			Mat estimated;

			double kalmanGaussianMean;
			double kalmanGaussianStdDev;
			double kalmanGaussianMixtureProportion;

			if (ksTest <= rainParams.maxGoFDifference) {

				Mat kalmanPredict = KF.predict();

				Mat measurement = (Mat_<double>(3, 1) <<
					gaussianMean, gaussianStdDev, gaussianMixtureProportion);
				estimated = KF.correct(measurement);

				kalmanGaussianMean = estimated.at<double>(0);
				kalmanGaussianStdDev = estimated.at<double>(1);
				kalmanGaussianMixtureProportion = estimated.at<double>(2);

				kalmanMeanTmp = kalmanGaussianMean;
				kalmanStdDevTmp = kalmanGaussianStdDev;
				kalmanMixPropTmp = kalmanGaussianMixtureProportion;

				if(rainParams.verbose)
					cout << "Updating Kalman filter" << endl;
			}
			else {

				kalmanGaussianMean = kalmanMeanTmp;
				kalmanGaussianStdDev = kalmanStdDevTmp;
				kalmanGaussianMixtureProportion = kalmanMixPropTmp;

				if (rainParams.verbose)
					cout << "Not updating Kalman filter" << endl;
			}

			if (rainParams.verbose) {
				cout << "EM Estimated: Mean: " << gaussianMean << ", std.dev: " <<
					gaussianStdDev << ", mix.prop: " << gaussianMixtureProportion << endl;
				cout << "Kalman:       Mean: " << kalmanGaussianMean << ", std.dev: " <<
					kalmanGaussianStdDev << ", mix.prop: " << kalmanGaussianMixtureProportion << endl;
			}


			if (rainParams.saveImg || rainParams.debug)
			{
				plotDistributions(histogram, gaussianMean, gaussianStdDev,
					gaussianMixtureProportion,
					kalmanGaussianMean, kalmanGaussianStdDev, kalmanGaussianMixtureProportion);
			}

			// 8. Detect the rain intensity from the mixture model
			// Now that we have estimated the distribution and the filtered distribution,
			// compute an estimate of the rain intensity
			// (Should only be calculated if kalmanGaussianMixtureProportion is a bove the threshold. Still calculated for plotting reasons)
				
			// Step 1: Compute the sum (surface) of the histogram
			double histSum = 0;

			for (auto& val : histogram) {
				histSum += val;
			}

			// Step 2: Compute the rain intensity R on both the estimate and filtered estimate
			double R = histSum * gaussianMixtureProportion;
			double kalmanR = histSum * kalmanGaussianMixtureProportion;

			resultsFile << 
				this->outputFolder + "/" + this->settingsFile + ";" +
				this->inputVideo + "; " +
				to_string(frameCounter) + "; " +
				to_string(gaussianMean) + ";" +
				to_string(gaussianStdDev) + ";" +
				to_string(gaussianMixtureProportion) + ";" +
				to_string(ksTest) + ";" +
				to_string(kalmanGaussianMean) + ";" +
				to_string(kalmanGaussianStdDev) + ";" +
				to_string(kalmanGaussianMixtureProportion) + ";" +
				to_string(R) + ";" +
				to_string(kalmanR) + "\n";

			if (rainParams.verbose)
				cout << "\n" << endl;
		}

	}
	resultsFile.close();
	return 0;
}

BossuRainParameters BossuRainIntensityMeasurer::loadParameters(std::string filePath)
{
	BossuRainParameters newParams = getDefaultParameters();

	FileStorage fs(filePath, FileStorage::READ);

	if (fs.isOpened()) {
		int tmpInt;
		fs["c"] >> tmpInt;
		if (tmpInt != 0) {
			newParams.c = tmpInt;
		}

		fs["minimumBlobSize"] >> tmpInt;
		if (tmpInt >= 0) {
			newParams.minimumBlobSize = tmpInt;
		}

		fs["maximumBlobSize"] >> tmpInt;
		if (tmpInt > 0) {
			newParams.maximumBlobSize = tmpInt;
		}

		float tmpFloat;
		fs["dm"] >> tmpFloat;
		if (tmpFloat > 0.) {
			newParams.dm = tmpFloat;
		}

		fs["maxGoFDifference"] >> tmpFloat;
		if (tmpFloat > 0.) {
			newParams.maxGoFDifference = tmpFloat;
		}

		fs["minimumGaussianSurface"] >> tmpFloat;
		if (tmpFloat > 0.) {
			newParams.minimumGaussianSurface = tmpFloat;
		}

		fs["emMaxIterations"] >> tmpInt;
		if (tmpInt > 0) {
			newParams.emMaxIterations = tmpInt;
		}


		fs["saveImg"] >> newParams.saveImg;
		fs["verbose"] >> newParams.verbose;
		fs["debug"] >> newParams.debug;
	}

	return newParams;
}

int BossuRainIntensityMeasurer::saveParameters(std::string filePath)
{
	FileStorage fs(filePath, FileStorage::WRITE);

	if (fs.isOpened()) {
		fs << "c" << rainParams.c;
		fs << "minimumBlobSize" << rainParams.minimumBlobSize;
		fs << "maximumBlobSize" << rainParams.maximumBlobSize;
		fs << "dm" << rainParams.dm;
		fs << "maxGoFDifference" << rainParams.maxGoFDifference;
		fs << "minimumGaussianSurface" << rainParams.minimumGaussianSurface;
		fs << "emMaxIterations" << rainParams.emMaxIterations;
		fs << "saveImg" << rainParams.saveImg;
		fs << "verbose" << rainParams.verbose;
		fs << "debug" << rainParams.debug;
	}
	else {
		return 1;
	}
}

BossuRainParameters BossuRainIntensityMeasurer::getDefaultParameters()
{
	BossuRainParameters defaultParams;

	defaultParams.c = 3;
	defaultParams.dm = 1.;
	defaultParams.emMaxIterations = 100;
	defaultParams.minimumBlobSize = 4;
	defaultParams.maximumBlobSize = 50;
	defaultParams.maxGoFDifference = 0.06;
	defaultParams.minimumGaussianSurface = 0.35;
	defaultParams.saveImg = true;
	defaultParams.verbose = true;
	defaultParams.debug = true;

	
	return defaultParams;
}

void BossuRainIntensityMeasurer::computeOrientationHistogram(
	const std::vector<std::vector<cv::Point>>& contours, 
	std::vector<double>& histogram, 
	double dm)
{
	// Compute the moments from the contours and get the orientation of the BLOB
	histogram.clear();
	histogram.resize(180);


	for (auto contour : contours) {
		Moments mu = moments(contour, false);
		
		// Compute the major semiaxis of the ellipse equivalent to the BLOB
		// In order to do so, we must compute eigenvalues of the matrix
		// | mu20 mu11 |
		// | mu11 mu02 |
		// Bossu et al, 2011, pp. 6, equation 16
		Mat momentsMat = Mat(2, 2, CV_64FC1);
		momentsMat.at<double>(0, 0) = mu.mu20;
		momentsMat.at<double>(1, 0) = mu.mu11;
		momentsMat.at<double>(0, 1) = mu.mu11;
		momentsMat.at<double>(1, 1) = mu.mu02;

		Mat eigenvalues;
		eigen(momentsMat, eigenvalues);

		// Extract the largest eigenvalue and compute the major semi-axis according to
		// Bossu et al, 2011, pp. 6, equation 14
		double a = 2 * sqrt(eigenvalues.at<double>(0, 0));
		a = a > 0 ? a : 1;

		// Bossu et al, 2011, pp. 6, equation 17
		double orientation = 0.5 * (atan2(2 * mu.mu11, (mu.mu02 - mu.mu20)));

		//Convert from [-\pi / 2, \pi /2] to [0, \pi]
		orientation += CV_PI / 2.;

		// Compute the uncertainty of the estimate to be used as standard deviation
		// for Gaussian
		double estimateUncertaintyNominator = sqrt(pow(mu.mu02 - mu.mu20, 2) + 2 * pow(mu.mu11, 2)) * dm;
		double estimateUncertaintyDenominator = pow(mu.mu02 - mu.mu20, 2) + 4 * pow(mu.mu11, 2);

		double estimateUncertainty = estimateUncertaintyDenominator > 0 ? 
			estimateUncertaintyNominator / estimateUncertaintyDenominator :
			1 ;

		if (rainParams.debug) {
			std::cout << "Orient (Deg): " << orientation * 180./CV_PI << ", unct: " << estimateUncertainty << ", m.semiaxis: " << a << endl;
		}

		// Compute the Gaussian (Parzen) estimate of the true orientation and 
		// add to the histogram
		for (double angle = 0; angle < histogram.size(); ++angle) {
			double angle_rad = angle * CV_PI / 180.;
			histogram[angle] += a / (estimateUncertainty * sqrt(2*CV_PI)) *
				exp(-0.5 * pow((angle_rad - orientation) / estimateUncertainty, 2));
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

	//Sort the histogram
	std::vector<double> histogram_sorted = histogram;
	std::sort(histogram_sorted.begin(), histogram_sorted.end());

	if(rainParams.debug)
		cout << "Largest histogram value: " << histogram_sorted[histogram_sorted.size() - 1] << endl;

	// Find the median value in the sorted histogram.
	double median = 0;

	if (histogram_sorted.size() % 2 == 0)
		median = (histogram_sorted[histogram_sorted.size() / 2 - 1] + histogram_sorted[histogram_sorted.size() / 2]) / 2;
	else
		median = histogram_sorted[histogram_sorted.size() / 2];

	// Now that we have found the median, only use entries in the histogram equal to or above
	// the median to calculate the mean, std.dev and proportion estimate
	// i.e. subtract median and only use postive values
	double sumAboveMedian = 0, observationSum = 0;
	double initialStdDev = 0;

	for (auto i = 0; i < histogram.size(); ++i) {
		double val = histogram[i] - median;
		if (val < 0.0)
			continue;

		sumAboveMedian += i * val;
		observationSum += val;
	}

	double initialMean = observationSum > 0 ? sumAboveMedian / observationSum : 90;

	double sumOfSqDiffToMean = 0;

	for (auto i = 0; i < histogram.size(); ++i) {
		double val = histogram[i] - median;
		if (val < 0.0)
			continue;

		sumOfSqDiffToMean += pow(i - initialMean, 2) * val;
	}

	initialStdDev = observationSum > 0 ? sqrt(sumOfSqDiffToMean / observationSum) : 0;

	// Use the observationSum / 180 to estimate the mixture proportion
	double uniformDistEstimate = 1. / histogram.size(); // if observationSum/180 result in value above 1, which then results in negative number when saying 1-unifDistEst
	double initialMixtureProportion = 0.;

	if(rainParams.debug)
		cout << "Median: " << median << ", SumAboveMean: " << sumAboveMedian << ", ObservationSum: " << observationSum << ", initialMean: " << initialMean << ", sumOfSquareDiff: " << sumOfSqDiffToMean << ", initialStdDev: " << initialStdDev << ", uniform Dist est.: " << uniformDistEstimate << endl;

	for (auto i = 0; i < histogram.size(); ++i) {
		double val = histogram[i] - median;

		initialMixtureProportion += val > 0 ? 
			((1. - uniformDistEstimate) * val) : 0;
	}

	initialMixtureProportion = observationSum > 0 ? initialMixtureProportion / observationSum : 0;

	// Now that we have the initial values, we may start the EM algorithm
	vector<double> estimatedMixtureProportion{ initialMixtureProportion };
	vector<double> estimatedGaussianMean{ initialMean };
	vector<double> estimatedGaussianStdDev{ initialStdDev };
	vector<double> z;
	double uniformMass = uniformDist(0, 180, 1);

	if (rainParams.verbose) {
		std::cout << "Mean: " << estimatedGaussianMean.back()
			<< ", stdDev: " << estimatedGaussianStdDev.back() << ", mixProp: " <<
			estimatedMixtureProportion.back() << endl;
	}


	for (auto i = 1; i < rainParams.emMaxIterations; ++i) {
		// Expectation step
		z.clear();

		for (double angle = 0; angle < 180.; ++angle) {
			z.push_back(((1. - estimatedMixtureProportion.back()) * uniformMass) /
				(estimatedMixtureProportion.back() * gaussianDist(estimatedGaussianMean.back(),
					estimatedGaussianStdDev.back(), angle) +
					(1. - estimatedMixtureProportion.back()) * uniformMass));
		}

		// Maximization step
		double meanNominator = 0;
		double meanDenominator = 0;
		double stdDevNominator = 0;
		double stdDevDenominator = 0;
		double mixtureProportionNominator = 0;
		double mixtureProportionDenominator = 0;

		for (double angle = 0; angle < 180; ++angle) {
			meanNominator += (1. - z[angle]) * angle * histogram[angle];
			meanDenominator += (1. - z[angle]) * histogram[angle];
		}
		double tmpGaussianMean = meanDenominator > 0 ? meanNominator / meanDenominator : 90.;
		estimatedGaussianMean.push_back(tmpGaussianMean);

		for (double angle = 0; angle < 180; ++angle) {
			stdDevNominator += ((1. - z[angle]) * pow(angle - estimatedGaussianMean.back(),2) *
				histogram[angle]);
			stdDevDenominator += (1. - z[angle]) * histogram[angle];

			mixtureProportionNominator += (1. - z[angle]) * histogram[angle];
			mixtureProportionDenominator += histogram[angle];
		}
		double tmpGaussianStdDev = stdDevDenominator > 0 ? sqrt(stdDevNominator / stdDevDenominator) : 0;
		estimatedGaussianStdDev.push_back(tmpGaussianStdDev);
		
		double tmpMixtureProportion = mixtureProportionDenominator > 0 ? mixtureProportionNominator /
			mixtureProportionDenominator : 0.;
		estimatedMixtureProportion.push_back(tmpMixtureProportion);

		if ((i % 25 == 0) && rainParams.verbose) {
			std::cout << "EM step " << i << ": Mean: " << estimatedGaussianMean.back()
				<< ", stdDev: " << estimatedGaussianStdDev.back() << ", mixProp: " <<
				estimatedMixtureProportion.back() << endl;
		}
	}

	gaussianMean = estimatedGaussianMean.back();
	gaussianStdDev = estimatedGaussianStdDev.back();
	gaussianMixtureProportion = estimatedMixtureProportion.back();
}

void BossuRainIntensityMeasurer::plotDistributions(const std::vector<double>& histogram, const double gaussianMean, const double gaussianStdDev, const double gaussianMixtureProportion, const double kalmanGaussianMean, const double kalmanGaussianStdDev, const double kalmanGaussianMixtureProportion)
{
	// Create canvas to plot on
	vector<Mat> channels;

	for (auto i = 0; i < 3; ++i) {
		channels.push_back(Mat::ones(300, 180, CV_8UC1) * 125);
	}
	
	Mat figure;
	cv::merge(channels, figure);
	
	// Plot histogram
	// Find maximum value of histogram to scale
	double maxVal = 0;
	for (auto &val : histogram) {
		if (val > maxVal) {
			maxVal = val;
		}
	}

	double scale = maxVal > 0 ? 300 / maxVal : 1;

	// Constrain the scale value based on the scale needed to display a uniform distribution
	// with gaussianMixtureProportion == 0
	double uniformScale = 100 / uniformDist(0, 180, 1);

	scale = uniformScale < scale ? uniformScale : scale;


	if (histogram.size() >= 180) {
		for (auto i = 0; i < 180; ++i) {
			line(figure, Point(i, 299), Point(i, 300 - std::round(histogram[i] * scale)), Scalar(255, 255, 255));
		}
	}

	// Plot estimated Gaussian, uniform, Kalman filtered Gaussian, uniform
	for (auto i = 0; i < 180; ++i) {
		double gaussianDensity = std::round(gaussianDist(gaussianMean, gaussianStdDev, i) * 
			gaussianMixtureProportion * scale);
		double uniformDensity = uniformDist(0, 180, i) * (1 - gaussianMixtureProportion) * scale;
		double estimatedDensity = gaussianDensity + uniformDensity;
		figure.at<Vec3b>(300 - estimatedDensity, i) = Vec3b(0, 0, 0); // Estimated density is black
		
		double kalmanGaussianDensity = std::round(gaussianDist(kalmanGaussianMean,
			kalmanGaussianStdDev, i) * kalmanGaussianMixtureProportion * scale);
		double kalmanUniformDensity = uniformDist(0, 180, i) *
			(1 - kalmanGaussianMixtureProportion) * scale;
		double kalmanEstimatedDensity = kalmanGaussianDensity + kalmanUniformDensity;
		figure.at<Vec3b>(300 - kalmanEstimatedDensity, i) = Vec3b(255, 240, 108); // Kalman estimate is cyan		
	}

	if(rainParams.debug)
		imshow("Histogram", figure);

	if (rainParams.saveImg) 
		imwrite("Histogram.png", figure);

}

double BossuRainIntensityMeasurer::goodnessOfFitTest(const std::vector<double>& histogram,
	const double gaussianMean,
	const double gaussianStdDev,
	const double gaussianMixtureProportion) {
	//Implementation of Goodness-Of-Fit / Kolmogrov-Smirnov test, pp. 8, equation 26

	//Calculate the sum of the unnormalized histogram
	double cummulativeSum = 0;
	for (auto& n : histogram) {
		cummulativeSum += n;
	}

	if (rainParams.debug)
		cout << "Cummulative sum of histogram: " << cummulativeSum << endl;

	//Calculate the Emperical CDF of the Histogram
	std::vector<double> eCDF(180, 0);

	//First determine the unnormalized bins
	eCDF[0] = histogram[0];
	for (auto i = 1; i < histogram.size(); ++i) {
		eCDF[i] = histogram[i] + eCDF[i - 1];
	}

	//Then normalize the bins
	for (auto i = 0; i < histogram.size(); ++i)
		eCDF[i] /= cummulativeSum;

	//Compare the Emperical CDF with the CDF of the actual joint unifrom-Gaussian distribution
	//Save the largest distance between the two CDFs
	double D = 0;
	for (auto i = 0; i < histogram.size(); ++i) {
		double normalCDF = 1. / 2. * (1. + erf((i - gaussianMean) / (gaussianStdDev*sqrt(2.))));
		double uniformCDF = (i + 1.) / histogram.size();
		double combinedCDF = gaussianMixtureProportion*normalCDF + (1 - gaussianMixtureProportion)*uniformCDF;
		double diff = abs(combinedCDF - eCDF[i]);

		if (rainParams.debug)
			cout << "ECDF: " << eCDF[i] << ", Gauss CDF: " << normalCDF << ", Uniform CDF: " << uniformCDF << ", combinedCDF: " << combinedCDF << ", diff: " << diff << endl;

		if (diff > D)
			D = diff;
	}
	if (rainParams.verbose)
		cout << "Goodness-Of-Fit test resulted in D: " << D << endl;

	return D;
}

void BossuRainIntensityMeasurer::plotGoodnessOfFitTest(const std::vector<double>& histogram,
	const double gaussianMean,
	const double gaussianStdDev,
	const double gaussianMixtureProportion) {
	//Plot Goodness-Of-Fit / Kolmogrov-Smirnov test, pp. 8, equation 26
	//Moved to separate function to avoid calculate plot if not desired

	//Calculate the sum of the unnormalized histogram
	double cummulativeSum = 0;
	for (auto& n : histogram) {
		cummulativeSum += n;
	}
	//Calculate the Emperical CDF of the Histogram
	std::vector<double> eCDF(180, 0);

	//First determine the unnormalized bins
	eCDF[0] = histogram[0];
	for (auto i = 1; i < histogram.size(); ++i) {
		eCDF[i] = histogram[i] + eCDF[i - 1];
	}

	//Then normalize the bins
	for (auto i = 0; i < histogram.size(); ++i)
		eCDF[i] /= cummulativeSum;

	//Plot the Emperical CDF and the CDF of the actual joint unifrom-Gaussian distribution

	// Create canvas to plot on
	vector<Mat> channels;

	for (auto i = 0; i < 3; ++i) {
		channels.push_back(Mat::ones(300, 180, CV_8UC1) * 125);
	}

	Mat ECDFFigure, uCDFFigure, nCDFFigure, cCDFFigure;
	cv::merge(channels, ECDFFigure);
	ECDFFigure.copyTo(uCDFFigure);
	ECDFFigure.copyTo(nCDFFigure);
	ECDFFigure.copyTo(cCDFFigure);
	double scale = 300;

	for (auto i = 0; i < histogram.size(); ++i) {
		double normalCDF = 1. / 2. * (1. + erf((i - gaussianMean) / (gaussianStdDev*sqrt(2.))));
		double uniformCDF = (i + 1.) / histogram.size();
		double combinedCDF = gaussianMixtureProportion*normalCDF + (1 - gaussianMixtureProportion)*uniformCDF;

		line(ECDFFigure, Point(i, 299), Point(i, 300 - std::round(eCDF[i] * scale)), Scalar(255, 255, 255));
		line(uCDFFigure, Point(i, 299), Point(i, 300 - std::round(uniformCDF * scale)), Scalar(255, 255, 255));
		line(nCDFFigure, Point(i, 299), Point(i, 300 - std::round(normalCDF * scale)), Scalar(255, 255, 255));
		line(cCDFFigure, Point(i, 299), Point(i, 300 - std::round(combinedCDF * scale)), Scalar(255, 255, 255));
	}

	if (rainParams.saveImg) {
		imwrite("ECDF.png", ECDFFigure);
		imwrite("Uniform CDF.png", uCDFFigure);
		imwrite("Normal CDF.png", nCDFFigure);
		imwrite("Combined CDF.png", cCDFFigure);
	}
	if (rainParams.debug) {
		imshow("ECDF", ECDFFigure);
		imshow("Uniform CDF", uCDFFigure);
		imshow("Normal CDF", nCDFFigure);
		imshow("Combined CDF", cCDFFigure);
	}
}

double BossuRainIntensityMeasurer::uniformDist(double a, double b, double pos)
{
	assert(b >= a);

	if (pos >= a && pos <= b) {
		return 1. / (b - a + 1);
	}

	return 0.;
}

double BossuRainIntensityMeasurer::gaussianDist(double mean, double stdDev, double pos)
{
	double result = 0;

	if (stdDev != 0) {
		result = 1 / sqrt(2 * CV_PI*pow(stdDev, 2)) *
			exp(-pow(pos - mean, 2) / (2 * pow(stdDev, 2)));
	}
	else if (mean == pos) {
		result = 1.;
	}

	return result;
}

int main(int argc, char** argv)
{
	const string keys =
		"{help h            |       | Print help message}"
		"{fileName fn       |       | Input video file to process}"
		"{filePath fp       |       | Filepath of the input file}"
		"{outputFolder of   |       | Output folder of processed files}"
		"{settingsFile sf   |       | File from which settings are loaded/saved}"
		"{saveSettings s    | 0     | Save settings to settingsFile (0,1)}"	
		"{saveImage i       | 0     | Save images from intermediate processing}"
		"{verbose v         | 0     | Write additional debug information to console}"
		"{debug d           | 0     | Enables debug mode. Writes extra information to console and shows intermediate images}"
		;

	cv::CommandLineParser cmd(argc, argv, keys);

	if (argc <= 1 || (cmd.has("help"))) {
		std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
		std::cout << "Available options: " << std::endl;
		cmd.printMessage();
		cmd.printErrors();
		std::cout << "Current arguments: " << endl;
		std::cout << **argv << endl;
		return 1;
	}

	for (int i = 0; i < argc; i++)
		cout << argv[i] << endl;
		
	std::string filename = cmd.get<std::string>("fileName");
	std::string filePath = cmd.get<std::string>("filePath");
	std::string outputFolder = cmd.get<string>("outputFolder");
	std::string settingsFile = cmd.get<string>("settingsFile");

	if ((settingsFile == "") && (cmd.get<int>("saveSettings") != 0))
		settingsFile = filename + "_" + "Settings.txt";

	BossuRainParameters defaultParams = BossuRainIntensityMeasurer::getDefaultParameters();

	if (cmd.has("settingsFile")) {
		defaultParams = BossuRainIntensityMeasurer::loadParameters(settingsFile);
	}

	// Set parameters here
	defaultParams.saveImg = cmd.get<int>("saveImage") != 0;
	defaultParams.verbose = cmd.get<int>("verbose") != 0;
	defaultParams.debug = cmd.get<int>("debug") != 0;

	BossuRainIntensityMeasurer bossuIntensityMeasurer(filename, filePath, settingsFile, outputFolder, defaultParams);

	// Save final settings
	if (cmd.get<int>("saveSettings") != 0) {
		bossuIntensityMeasurer.saveParameters(outputFolder + "/" + settingsFile);
	}

	bossuIntensityMeasurer.detectRain();


}
 