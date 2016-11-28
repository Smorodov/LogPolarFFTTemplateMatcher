#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
//-----------------------------------------------------------------------------------------------------
// As input we need equal sized images, with the same aspect ratio,
// scale difference should not exceed 1.8 times.
//-----------------------------------------------------------------------------------------------------
cv::RotatedRect LogPolarFFTTemplateMatch(cv::Mat& im0, cv::Mat& im1);