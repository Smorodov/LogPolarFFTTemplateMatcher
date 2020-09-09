#include "fftm.hpp"
using namespace std;
using namespace cv;
//-----------------------------------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------------------------------
int main(int argc, unsigned int** argv)
{
    Mat im0 = imread("cat.png", 1);
    Mat im1 = imread("cat_part.png", 1);

    imshow("im1", im1);
    imshow("im0", im0);

    // As input we need equal sized images, with the same aspect ratio,
    // scale difference should not exceed 1.8 times.

    RotatedRect rr = LogPolarFFTTemplateMatch(im0, im1,200,100);

    // Plot rotated rectangle, to check result correctness
    Point2f rect_points[4];
    rr.points(rect_points);
    for (int j = 0; j < 4; j++)
    {
        line(im0, rect_points[j], rect_points[(j + 1) % 4], Scalar(255, 0, 0), 2, LINE_AA);
    }

    imshow("result", im0);

    waitKey();
}
