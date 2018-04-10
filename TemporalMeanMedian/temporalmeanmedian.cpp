#include "temporalmeanmedian.h"

using namespace std;
using namespace cv;

TemporalMeanMedian::TemporalMeanMedian(std::string inputVideo, std::string outputFolder, int temporalN)
{
	this->inputVideo = inputVideo;
	this->outputFolder = outputFolder;
	this->temporalN = temporalN;
}

int TemporalMeanMedian::performFiltering()
{
	VideoCapture cap(inputVideo);
	int imageNbr = 0;

	if (cap.isOpened()) {
		cout << "Opening video: " << inputVideo << endl;

		std::vector<std::pair<int, cv::Mat>> sourceImages;
		sourceImages.resize(temporalN);


		for (auto i = 0; i < temporalN; ++i) {
			Mat image;
			cap >> image;
			sourceImages[i] = make_pair(imageNbr, image);

			++imageNbr;
		}
		string currentOutput = outputFolder + "/TemporalMean";

		CreateDirectoryA(currentOutput.c_str(), NULL);

		while (cap.grab()) {
			Mat image;
			cap >> image;

			cv::Mat temporalMean = meanFilter(sourceImages);
			stringstream outFrameNumber;
			int meanImageNumber = imageNbr - (temporalN / 2 - 1);
			outFrameNumber << setw(5) << setfill('0') << meanImageNumber;
			imwrite(currentOutput + '/' +outFrameNumber.str() + ".png", temporalMean);

			// Bookkeeping of previous images
			for (auto i = (temporalN - 1); i > 0; --i) {
				sourceImages[i - 1].first = sourceImages[i].first;
				sourceImages[i - 1].second = sourceImages[i].second.clone();
			}

			sourceImages.back().second = image;
			sourceImages.back().first = imageNbr;



			++imageNbr;
		}
	}

	return 0;
}

cv::Mat TemporalMeanMedian::medianFilter(const std::vector<std::pair<int, cv::Mat>>& frames)
{
	if (frames.size() < 3) {
		return Mat();
	}

	Mat median(frames.front().second.size(), frames.front().second.type());

	for (auto i = 0; i < frames.size(); ++i) {
		if (i == (frames.size() / 2 + 1)) {
			continue;
		}

		//accumulate(frames[i].second, acc); Not fully implemented
	}
	
	
	return cv::Mat();
}

cv::Mat TemporalMeanMedian::meanFilter(const std::vector<std::pair<int, cv::Mat>>& frames)
{
	if (frames.size() < 3) {
		return Mat();
	}

	Mat acc(frames.front().second.size(), CV_64FC3);
	
	for (auto i = 0; i < frames.size(); ++i) {
		if (i == (frames.size() / 2)) {
			continue;
		}

		std::string windowName = "Input " + to_string(i);
		cv::imshow(windowName, frames[i].second);
		accumulate(frames[i].second, acc);
	}

	Mat avg;
	acc.convertTo(avg, CV_8UC3, 1. / (frames.size() - 1));
	cv::imshow("Mean", avg);
	cv::waitKey(0);
	
	return avg;
}



int main(int argc, char** argv)
{
	const string keys =
		"{help h            |       | Print help message}"
		"{fileName fn       |       | Input video file}"
		"{outputFolder of   |       | Output folder of processed files}"
		"{temporalN n       |       | Number of frames to compute median/mean}"
		;

	cv::CommandLineParser cmd(argc, argv, keys);

	if (argc <= 1 || (cmd.has("help"))) {
		std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
		std::cout << "Available options:" << std::endl;
		cmd.printMessage();
		return 1;
	}

	std::string filename = cmd.get<std::string>("fileName");
	std::string outputFolder = cmd.get<std::string>("outputFolder");
	int temporalN = cmd.get<int>("temporalN");

	TemporalMeanMedian meanMedian(filename, outputFolder, temporalN);
	meanMedian.performFiltering();
}