/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - houghDart.cpp
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
void ddx(cv::Mat &input, cv::Mat &resultX, cv::Mat &resultY);
void magnitude(cv::Mat &input, cv::Mat &ddx, cv::Mat &ddy, cv::Mat &magnitude);
void direction(cv::Mat &input, cv::Mat &ddx, cv::Mat &ddy, cv::Mat &direction);
void thresholding(Mat &input, Mat &thresholded);
void hough( Mat tgMagImage, int threshold);


/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;

/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	// cv::Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	cv::Mat frame = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	// detectAndDisplay( frame );
	// start of sobel
	cv::Mat ddx1;
	cv::Mat ddy;
	cv::Mat mag;
	cv::Mat dir;
	cv::Mat thresh; 

	// Compute image containing derivative in x direction and y direction
	ddx(frame, ddx1, ddy);	
		
	// magnitude of gradient
	magnitude(frame, ddx1, ddy, mag);

	// direction of gradient	
	direction(frame, ddx1, ddy, dir);

	// WHAT DOES IT PRODUCE? --> result
	// thresholding the gradient magnitude image
	thresholding(mag, thresh);
		
	// then hough()


	// 4. Save Result Image
	imwrite( "sobel.jpg", frame );
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
	double kernelXInit[3][3] = {{ -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 }};
	cv::Mat kernel_X = cv::Mat(size, size, CV_32FC2, kernelXInit);

	double kernelYInit[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
	cv::Mat kernel_Y = cv::Mat(size, size, CV_32FC2, kernelYInit);
	
	// create padded version of input to prevent border effects
	int kernelXRad = (kernel_X.size[0] -1) / 2;
	int kernelYRad = (kernel_Y.size[1] -1) / 2;

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
					double kernelValX = kernel_X.at<double> (kernelX, kernelY);  
				 	double kernelValY = kernel_Y.at<double> (kernelX, kernelY); 
					sumX += inputVal * kernelValX;
					sumY += inputVal * kernelValY;
				}

			}	
			resultX.at<double>(i,j) = sumX;			
			resultY.at<double>(i,j) = sumY;
		}
	}	
}

// magnitude of gradient for sobel
void magnitude (cv::Mat &input, cv::Mat &ddx, cv::Mat &ddy, cv::Mat &magnitude) {
	// Get the derivatices first and use them for the magnitude
	// cv::Mat ddx1;
	// cv::Mat ddy1;
	// ddx(input, ddx1, ddy1);

	cout<< "hereeeeeeeeeeeeeeeeeee" << endl;

	magnitude = cv::Mat (input.rows, input.cols, CV_32FC2);
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			magnitude.at<double>(i,j) = sqrt( pow( ddx.at<double> (i,j), 2) + pow( ddy.at<double> (i,j), 2));	
		}
	}	
}

// what this do
void direction (cv::Mat &input, cv::Mat &ddx, cv::Mat &ddy, cv::Mat &direction){
	direction = cv::Mat (input.rows, input.cols, CV_32FC2);
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++ ) {
			direction.at<double>(i,j) = atan( ddy.at<double> (i,j) / ddx.at<double> (i,j));
		}
	}
}

void thresholding( Mat &input, Mat &thresholded){
		cout<< "also hereeeeee" << endl;

	// take gradient magnitude and apply thresholding operation
	// -> to get set of pixels with trongest g. magnitude to be considered for circle detection
	thresholded = cv::Mat (input.rows, input.cols, CV_32FC2);

	// how to go through values of pixelsssss
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++ ) {
			if (input.at<double>(i,j) > 105 ) {
				thresholded.at<double>(i,j) = 255;
			}
			else {
				thresholded.at<double>(i,j) = 0;
			}
		}
	}

}


// ---> min/max radius and distance between circle centres
void hough( Mat tgMagImage, int threshold ){
	int houghSpace[5][5][5]; //????????

	// calculates 3D hough space (xo, yo, r) from threshold g. mag. image and gradient orientation image
	// decide size (no of cells) in hough space


	// display hough space for each image
	// --> create a 2D image
	// ---> take log of the image to make values more descriptive


	// threshold the hough space and display the set of found circles on og images 
}



/** @function detectAndDisplay **/
// THIS NEEDS TO BE CHANGED SO THAT IT WORKS ON DARTBOARDS NOT FACES 
void detectAndDisplay( Mat frame ){
	// std::vector<Rect> faces;
	
	Mat frame_gray;
	// Mat result;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	// cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
	// sobel (frame_gray, result);
	
	// imwrite("sobel.jpg", result);
       // 3. Print number of Faces found
	// std::cout << faces.size() << std::endl;

       // 4. Draw box around faces found
	/* for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}*/

}



