************************
LogPolarFFTTemplateMatch
************************

LogPolarFFTTemplateMatch
------------------------

:: 

    RotatedRect LogPolarFFTTemplateMatch(Mat& im0, Mat& im1, double canny_threshold1, double canny_threshold2)

Arguments
*********
    * im0 - first input image
    * im1 - second input image
    * canny_threshold1 - thresholding parameter for Canny edge detector used for input image preprocessing.
    * canny_threshold2 - thresholding parameter for Canny edge detector used for input image preprocessing.

.. note:: Images im0 and im1 must have the same size, and must have the same aspect ratio.

Return value
************
    * Returns registration result as RotatedRect, if method fails it will return RotatedRect filled with all zeros.

Short description
*****************

An FFT-based technique for translation, rotation and scale-invariant image registration.

Usage example
*************
:: 

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
            line(im0, rect_points[j], rect_points[(j + 1) % 4], Scalar(255, 0, 0), 2, CV_AA);
        }

        imshow("result", im0);

        waitKey();
    }

Result should look like image below: 

.. image:: ../FFTTM.PNG
    :width: 400pt

References
**********

(1) An FFT-based technique for translation, rotation and scale-invariant
    image registration. BS Reddy, BN Chatterji.
    IEEE Transactions on Image Processing, 5, 1266-1271, 1996
    
(2) An IDL/ENVI implementation of the FFT-based algorithm for automatic
    image registration. H Xiea, N Hicksa, GR Kellera, H Huangb, V Kreinovich.
    Computers & Geosciences, 29, 1045-1055, 2003.
    
(3) Image Registration Using Adaptive Polar Transform. R Matungka, YF Zheng,
    RL Ewing. IEEE Transactions on Image Processing, 18(10), 2009.
    
Based on python version of algorithm:  http://www.lfd.uci.edu/~gohlke/code/imreg.py.html 