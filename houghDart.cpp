/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


/** Function Headers */
void detectAndDisplay( Mat frame );
void sobel( Mat frame );
/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	// detectAndDisplay( frame );
	sobel( frame);

	// 4. Save Result Image
	//imwrite( "detectedDarts.jpg", frame );
;
	return 0;
}

// derivation in x direction
void ddx(cv::Mat &input, cv::Mat &resultX, cv::Mat &resultY ) {
	// wtf is this size - 
	int size = 3;

	// need to initialise the output back to sobel
	resultX = cv::Mat(input.rows, input.cols, CV_32FC2);
	resultY = cv::Mat(input.rows, input.cols, CV_32FC2);

	// we need a kernel - initialise kernel
	// this hsould be Gaussian kernal idk
	// do we want it an in tidk
	// int kernel[size][size];
	int kernelXInit[3][3] = {{ -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 }};
	cv::Mat kernel = cv::Mat(size, size, CV_32FC2, kernelXInit);

	int kernelYInit[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
	cv::Mat kernelY = cv::Mat(size, size, CV_32FC2, kernelYInit);
	
	// create padded version of input to prevent border effects
	int kernelXRad = (kernel.size[0] -1) / 2;
	int kernelYRad = (kernel.size[1] -1) / 2;

	// wtf is a padded input
	cv::Mat paddedInput;
	// make image with border
	cv::copyMakeBorder(input, paddedInput, kernelXRad, kernelXRad, kernelYRad, kernelYRad, cv::BORDER_REPLICATE);

	// Convolution shit
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			double sumX = 0.0;
			double sumY = 0.0;
			for (int m = -kernelXRad; m <= kernelXRad; m++){
				for (int n = -kernelYRad; n <= kernelYRad; n++){
					// indices
					int inputX = i + m + kernelXRad;
					int inputY = j + n + kernelYRad;
					int kernelX = m + kernelXRad;
					int kernelY = n + kernelYRad;

					// get values
					int inputVal = (int)paddedInput.at<uchar> (inputX, inputY);
					double kernelVal  = kernel.at<double> (kernelX, kernelY);  
					
					sum += inputVal * kernelVal;

				}

			}	
			resultX.at<double>(i,j) = sum;			
		}
	}	
}

// sobel function for first part of hough
void sobel ( Mat frame ) {
	// make greyscale first
	Mat frame_gray;
	cvtColor ( frame, frame_gray, CV_BGR2GRAY);
	// Compute image containing derivative in x direction
        cv::Mat ddx1;
	cv::Mat ddy2;
	ddx(frame_gray, ddx1, ddy1);	
	// y direction
	// magnitude of gradient
	// direction of gradient	

}


/** @function detectAndDisplay **/
// THIS NEEDS TO BE CHANGED SO THAT IT WORKS ON DARTBOARDS NOT FACES 
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

       // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

}
