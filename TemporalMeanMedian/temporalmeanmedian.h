#ifndef TEMPORALMEANMEDIAN_H
#define TEMPORALMEANMEDIAN_H

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2\opencv.hpp>
#include <Windows.h>


class TemporalMeanMedian
{
public:
	TemporalMeanMedian(std::string inputVideo, std::string outputFolder, int temporalN);

	int performFiltering();

private:
	cv::Mat medianFilter(const std::vector<std::pair<int, cv::Mat> >& frames);
	cv::Mat meanFilter(const std::vector<std::pair<int, cv::Mat> >& frames);

	std::string inputVideo, outputFolder;
	int temporalN;


};

#endif // !TEMPORALMEANMEDIAN_H

