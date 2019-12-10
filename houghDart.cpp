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
void sobelOpenCV( Mat &input, Mat &sobel );
void sobelDetection( Mat &input, Mat &dx, Mat &dy, Mat &mag, Mat &dir, Mat &sobel);

void detectAndDisplay( Mat frame );
void gaussianBlurFilter( Mat &input, Mat &gblur );
void ddx(cv::Mat &input, cv::Mat &resultX, cv::Mat &resultY);
void magnitude(cv::Mat &input, cv::Mat &ddx, cv::Mat &ddy, cv::Mat &magnitude);
void direction(cv::Mat &input, cv::Mat &ddx, cv::Mat &ddy, cv::Mat &direction);
void thresholding(Mat &input, Mat &thresholded);
void hough( Mat tgMagImage, int threshold);


/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;

/** @function main */
int main( int argc, const char** argv ){
	// 1. Read Input Image
	// cv::Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	cv::Mat frame = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	// detectAndDisplay( frame );

	// // thresholding the gradient magnitude image
	// thresholding(mag, thresh);
		

//-------------------------------------------------------------------------
	cv::Mat dx, dy, mag, dir, thresh, gblur, sobe, so;

	sobelOpenCV(frame, so);
	sobelDetection(frame, dx, dy, mag, dir, sobe);

//-------------------------------------------------------------------------


	// 4. Save Result Image
	imwrite( "helpme.jpg", sobe );
	//imwrite( "mag.jpg", dx );

	return 0;
}


void gaussianBlurFilter( Mat &input, Mat &gblur ){
	// this hsould be Gaussian kernal idk
	
	// use a gaussian blur first to get rid of the kow freq stuff noisee
	//or not ??

	// literally just convolving with a small gaussian kernel
}

void sobelOpenCV( Mat &input, Mat &sobel ){
  sobel = cv::Mat (input.rows, input.cols, CV_32FC1);

  /// Generate grad_x and grad_y
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  /// Gradient X
  Sobel( input, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );

  	imwrite( "sobelX.jpg", abs_grad_x );


  /// Gradient Y
  Sobel( input, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y );

	imwrite( "sobelY.jpg", abs_grad_y );


  /// Total Gradient (approximate)
  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobel ); // ---> also might be missing this?
															// this is the averaging part i think??
															// need to scale or the values mean nothing????

}


void sobelDetection( Mat &input, Mat &dx, Mat &dy, Mat &mag, Mat &dir, Mat &sobe){

	Mat abs_grad_x, abs_grad_y;
	sobe.create(input.size(), input.type());
	mag.create(input.size(), input.type());

	  		//GaussianBlur( input, input, Size(3,3), 0, 0, BORDER_DEFAULT ); -> makes it only slightly better

	// Compute image containing derivative in x direction and y direction
	ddx(input, dx, dy);
 	 convertScaleAbs( dx, abs_grad_x );
  	 convertScaleAbs( dy, abs_grad_y );
  	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobe ); // need this to combine result x and result y


	imwrite( "mag.jpg", abs_grad_y );

	// magnitude of gradient
	// magnitude(sobe, abs_grad_x, abs_grad_y, mag);

	// // direction of gradient
	// direction(input, dx, dy, dir);
	
	//sobe.convertTo(sobe, CV_8UC1);
	//mag.convertTo(mag, CV_8UC1);

}


// derivation in x direction
void ddx(cv::Mat &input, cv::Mat &resultX, cv::Mat &resultY ) {
	int size = 3; // size of the kernel matrix

	// need to initialise the output back to sobel
	resultX.create(input.size(), input.type());
	resultY.create(input.size(), input.type());

	// initialise kernels
	cv::Mat kernel_X = (Mat_<float>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    		//cout << "KX is = " << endl << " " << kernel_X << endl << endl;
	cv::Mat kernel_Y = (Mat_<float>(3,3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
	// -1, -2, -1, 0, 0, 0, 1, 2, 1 ---> this is better somehow??
    		// cout << "KY is = " << endl << " " << kernel_Y << endl << endl;

	// create padded version of input to prevent border effects
	int kernelXRad = (kernel_X.size[0] - 1) / 2;
	int kernelYRad = (kernel_Y.size[1] - 1) / 2;

	// wtf is a padded input lol
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
			resultX.at<uchar>(i,j) = (uchar) sumX;			
			resultY.at<uchar>(i,j) = (uchar) sumY;
		}
	}	
}

// magnitude of gradient for sobel
// pythagras theorem!!!!!!!!!!!!!!! SSS HH OOO KK
// how big is the edge at this direction
// scale it??
void magnitude (cv::Mat &input, cv::Mat &dx, cv::Mat &dy, cv::Mat &mag) {
	mag.create(input.size(), input.type());
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			mag.at<double>(i,j) = sqrt( pow(dx.at<double> (i,j), 2) + pow(dy.at<double> (i,j), 2));	
		}
	}
}

// what this do
void direction (cv::Mat &input, cv::Mat &ddx, cv::Mat &ddy, cv::Mat &direction){
	direction = cv::Mat (input.rows, input.cols, CV_32FC1);
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
	thresholded = cv::Mat (input.rows, input.cols, CV_32FC1);

	// how to go through values of pixelsssss
	// for (int i = 0; i < input.rows; i++) {
	// 	for (int j = 0; j < input.cols; j++ ) {
	// 		if (input.at<double>(i,j) > 105 ) {
	// 			thresholded.at<double>(i,j) = 255;
	// 		}
	// 		else {
	// 			thresholded.at<double>(i,j) = 0;
	// 		}
	// 	}
	// }

	cv::threshold(input, thresholded, 150, 0, THRESH_BINARY);

}


// ---> min/max radius and distance between circle centres
void hough( Mat tgMagImage, int threshold ){
	int houghSpace[5][5][5]; //????????
	vector<Vec3f> output;

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
	/* for( int i = 0; i < faces.size(); i++ ){
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}*/

}



