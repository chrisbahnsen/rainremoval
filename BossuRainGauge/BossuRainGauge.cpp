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

	// Create file, write header
	ofstream resultsFile;
	resultsFile.open(this->outputFolder + "/" + "bossuRainGauge.txt", ios::out | ios::trunc);

	std::string header;

	for (auto dmVal : rainParams.dm) {
		header += to_string(dmVal) + ";" + to_string(dmVal) + ";"; // We log the estimated rain with and without the Kalman filter
	}

	resultsFile << header + "\n";

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
		
		while (cap.grab()) {
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
			// to filter out large connected components
			vector<vector<Point> > contours, filteredContours; 

			if (rainParams.saveDebugImg) {
				imshow("Image", frame);
				imwrite("backgroundImg.png", backgroundImage);
				imshow("Foreground image", grayForegroundImage);
				imwrite("foregroundImage.png", grayForegroundImage);
				imshow("Difference image", diffImg);
				imwrite("diffImg.png", diffImg);
				imshow("Candidate rain pixels", candidateRainMask);
				imwrite("Candidate.png", candidateRainMask);

				waitKey(0);

				cout << "Sum of non-zero in candidateRainMask" << cv::countNonZero(candidateRainMask) << endl;
			}

			findContours(candidateRainMask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

			for (auto dmVal : rainParams.dm) {
				Mat filteredCandidateMask = candidateRainMask.clone();
				int deletedContours = 0;

				for (auto contour : contours) {
					if ((contour.size() > rainParams.maximumBlobSize) ||
						(contour.size() < rainParams.minimumBlobSize)) {
						
						for (auto point : contour) {
							filteredCandidateMask.at<uchar>(point.y, point.x) = 0;
						}
						deletedContours++;
					}
					else {
						// Retain the contour if below or equal to the size threshold
						filteredContours.push_back(contour);
					}
				}

				// Delete the contour from the rain image
				if (rainParams.verbose) {
					std::cout << "Deleted " << deletedContours << " contours with dm: " << dmVal << endl;
				}

				if (rainParams.saveDebugImg) {
					// For now, only show the debug images with this settings
					imshow("Filtered rain pixels", filteredCandidateMask);
				}

				// 4. Compute the Histogram of Orientation of Streaks (HOS) from the contours
				vector<double> histogram;
				computeOrientationHistogram(filteredContours, histogram, dmVal);

				// 5. Model the accumulated histogram using a mixture distribution of 1 Gaussian
				//    and 1 uniform distribution in the range [0-179] degrees.
				double gaussianMean, gaussianStdDev, gaussianMixtureProportion;

				estimateGaussianUniformMixtureDistribution(histogram, filteredContours.size(),
					gaussianMean, gaussianStdDev, gaussianMixtureProportion);

				// 6. Use a Kalman filter for each of the three parameters of the mixture
				//	  distribution to smooth the model temporally
				Mat kalmanPredict = KF.predict();

				Mat measurement = (Mat_<double>(3, 1) <<
					gaussianMean, gaussianStdDev, gaussianMixtureProportion);
				Mat estimated = KF.correct(measurement);

				double kalmanGaussianMean = estimated.at<double>(0);
				double kalmanGaussianStdDev = estimated.at<double>(1);
				double kalmanGaussianMixtureProportion = estimated.at<double>(2);

				if (rainParams.verbose) {
					cout << "EM Estimated: Mean: " << gaussianMean << ", std.dev: " <<
						gaussianStdDev << ", mix.prop: " << gaussianMixtureProportion << endl;
					cout << "Kalman:       Mean: " << kalmanGaussianMean << ", std.dev: " <<
						kalmanGaussianStdDev << ", mix.prop: " << kalmanGaussianMixtureProportion << endl;
				}

				if (rainParams.saveDebugImg)
				{
					plotDistributions(histogram, gaussianMean, gaussianStdDev,
						gaussianMixtureProportion,
						kalmanGaussianMean, kalmanGaussianStdDev, kalmanGaussianMixtureProportion);
				}

				// 7. Detect the rain intensity from the mixture model
				// Now that we have estimated the distribution and the filtered distribution,
				// compute an estimate of the rain intensity
				// Step 1: Compute the sum (surface) of the histogram
				double histSum = 0;

				for (auto& val : histogram) {
					histSum += val;
				}

				// Step 2: Compute the rain intensity R on both the estimate and filtered estimate
				double R = histSum * gaussianMixtureProportion;
				double kalmanR = histSum * kalmanGaussianMixtureProportion;

				resultsFile << to_string(R) + ";" + to_string(kalmanR) + ";\n";
			}
		}

	}

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

		std::vector<double> dm;
		fs["dm"] >> dm;
		if (!dm.empty()) {
			newParams.dm = dm;
		}

		fs["emMaxIterations"] >> tmpInt;
		if (tmpInt > 0) {
			newParams.emMaxIterations = tmpInt;
		}


		fs["saveDebugImg"] >> newParams.saveDebugImg;
		fs["verbose"] >> newParams.verbose;
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
		fs << "emMaxIterations" << rainParams.emMaxIterations;
		fs << "saveDebugImg" << rainParams.saveDebugImg;
		fs << "verbose" << rainParams.verbose;
	}
	else {
		return 1;
	}
}

BossuRainParameters BossuRainIntensityMeasurer::getDefaultParameters()
{
	BossuRainParameters defaultParams;

	defaultParams.c = 3;
	defaultParams.dm = { 1. };
	defaultParams.emMaxIterations = 100;
	defaultParams.minimumBlobSize = 4;
	defaultParams.maximumBlobSize = 50;
	defaultParams.saveDebugImg = true;

	
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
		a = a > 0 ? a : 1;

		// Bossu et al, 2011, pp. 6, equation 17
		double orientation = 0.5 * atan2(2 * mu.m11, (mu.m02 - mu.m20));

		// Convert to degrees and scale in range [0, \pi] ( [0 180] )
		orientation = orientation * 180. / CV_PI + 90;

		// Compute the uncertainty of the estimate to be used as standard deviation
		// for Gaussian
		double estimateUncertaintyNominator = sqrt(pow(mu.m02 - mu.m20, 2) + 2 * pow(mu.m11, 2)) * dm;
		double estimateUncertaintyDenominator = pow(mu.m02 - mu.m20, 2) + 4 * pow(mu.m11, 2);

		double estimateUncertainty = estimateUncertaintyDenominator > 0 ? 
			estimateUncertaintyNominator / estimateUncertaintyDenominator :
			1 ;

		if (rainParams.verbose) {
			std::cout << "Orient: " << orientation << ", unct: " 
				<< estimateUncertainty << ", m.semiaxis: " << a << endl;
		}

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
	double median = 0;

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
	double totalSum = 0;


	for (auto i = median; i < histogram.size(); ++i) {
		sumAboveMedian += i * histogram[i];
		observationSum += histogram[i];
	}

	double initialMean = observationSum > 0 ? sumAboveMedian / observationSum : 90;

	double sumOfSqDiffToMean = 0;

	for (auto i = median; i < histogram.size(); ++i) {
		sumOfSqDiffToMean += pow(i * histogram[i] - initialMean, 2);
	}

	initialStdDev = observationSum > 0 ? sqrt(sumOfSqDiffToMean / observationSum) : 0;

	// Use the observationSum / 180 to estimate the mixture proportion
	double uniformDistEstimate = observationSum / histogram.size(); 
	double initialMixtureProportion = 0;

	for (auto i = median; i < histogram.size(); ++i) {
		initialMixtureProportion += histogram[i] > 0 ? 
			(1 - (uniformDistEstimate / histogram[i]) * histogram[i]) : 0;
	}

	initialMixtureProportion = observationSum > 0 ? initialMixtureProportion / observationSum : 0;

	// Now that we have the initial values, we may start the EM algorithm
	vector<double> estimatedMixtureProportion{ initialMixtureProportion };
	vector<double> estimatedGaussianMean{ initialMean };
	vector<double> estimatedGaussianStdDev{ initialStdDev };
	vector<double> z;
	double uniformMass = uniformDist(0, 180, 1);


	for (auto i = 1; i < rainParams.emMaxIterations; ++i) {
		// Expectation step
		z.clear();

		for (double angle = 0; angle < 180.; ++angle) {
			z.push_back(((1. - estimatedMixtureProportion.back()) * uniformMass) /
				(estimatedMixtureProportion.back() * gaussianDist(estimatedGaussianMean.back(),
					estimatedGaussianStdDev.back(), angle) +
					(1. - estimatedMixtureProportion.back()) * uniformMass));

			double nominator = ((1. - estimatedMixtureProportion.back()) * uniformMass);
			double denominator = (estimatedMixtureProportion.back() * gaussianDist(estimatedGaussianMean.back(),
				estimatedGaussianStdDev.back(), angle) +
				(1. - estimatedMixtureProportion.back()) * uniformMass);

			int k = 2 + denominator;
		}

		// Maximization step
		double meanNominator = 0;
		double meanDenominator = 0;
		double stdDevNominator = 0;
		double stdDevDenominator = 0;
		double mixtureProportionNominator = 0;
		double mixtureProportionDenominator = 0;

		for (double angle = 0; angle < 180; ++angle) {
			meanNominator += (1 - z[angle]) * angle * histogram[angle];
			meanDenominator += (1 - z[angle]) * histogram[angle];
		}
		double tmpGaussianMean = meanDenominator > 0 ? meanNominator / meanDenominator : 90.;
		estimatedGaussianMean.push_back(tmpGaussianMean);

		for (double angle = 0; angle < 180; ++angle) {
			stdDevNominator += ((1 - z[angle]) * (angle - estimatedGaussianMean.back()) *
				histogram[angle]);
			stdDevDenominator += (1 - z[angle]) * histogram[angle];

			mixtureProportionNominator += (1 - z[angle]) * histogram[angle];
			mixtureProportionDenominator += histogram[angle];
		}
		double tmpGaussianStdDev = stdDevDenominator > 0 ? stdDevNominator / stdDevDenominator : 0;
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

	imshow("Histogram", figure);
}

double BossuRainIntensityMeasurer::uniformDist(double a, double b, double pos)
{
	if (pos >= a || pos >= b) {
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
		"{outputFolder of   |       | Output folder of processed files}"
		"{settingsFile sf   |       | File from which settings are loaded/saved}"
		"{saveSettings s    | 0     | Save settings to settingsFile (0,1)}"	
		"{debugImage di     | 0     | Save debug images from intermediate processing}"
		"{verbose v         | 0     | Write additional debug information to console}"
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

	std::string filename = cmd.get<std::string>("fileName");
	std::string outputFolder = cmd.get<string>("outputFolder");

	BossuRainParameters defaultParams = BossuRainIntensityMeasurer::getDefaultParameters();

	if (cmd.has("settingsFile")) {
		defaultParams = BossuRainIntensityMeasurer::loadParameters(cmd.get<std::string>("settingsFile"));
	}

	// Set parameters here
	defaultParams.saveDebugImg = cmd.get<int>("debugImage") != 0;
	defaultParams.verbose = cmd.get<int>("verbose") != 0;

	BossuRainIntensityMeasurer bossuIntensityMeasurer(filename, outputFolder, defaultParams);

	// Save final settings
	if (cmd.has("settingsFile") && (cmd.get<int>("saveSettings") != 0)) {
		bossuIntensityMeasurer.saveParameters(cmd.get<std::string>("settingsFile"));
	}

	bossuIntensityMeasurer.detectRain();


}
 