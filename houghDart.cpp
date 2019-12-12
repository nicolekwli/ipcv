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

void sobelDetection( Mat &input, Mat &dx, Mat &dy, Mat &mag, Mat &dir, Mat &sobel);
void ddx(cv::Mat &input, cv::Mat &resultX, cv::Mat &resultY);
void magnitude(cv::Mat &input, cv::Mat &ddx, cv::Mat &ddy, cv::Mat &magnitude);
void direction(cv::Mat &input, cv::Mat &ddx, cv::Mat &ddy, cv::Mat &direction);

void thresholding(Mat &input, Mat &thresholded);
void hough(Mat frame, Mat &mag, Mat &dir, int thresh, int peak, int maxR, int minR);
vector<cv::Vec3i> houghCircleDetection( Mat &mag, Mat &dir, int thresh, int peak, int maxR, int minR);
void scaling(int *** hough, int maxR, int cols, int rows);


/** Global variables */
String cascade_name = "/dartcascade/cascade.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv ){
	// 1. Read Input Image
	cv::Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	cv::Mat frame_gray = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	// detectAndDisplay( frame );

	// declaring all the matrices here
	cv::Mat dx, dy, mag, dir, thresh, gblur, sobe, so;

	//sobelOpenCV(frame, so);
	sobelDetection(frame_gray, dx, dy, mag, dir, sobe);

	// thresholding the gradient magnitude image
	// DO NOT need to pass this into hough()
	thresholding(mag, thresh);

	/*	params:
			- the original image so we can draw circles
			- magnitude matrix
			- gradient direction matrix
			- threshold that is used in the thresholding function 
			- peak -> no. of votes greater than which we consider that point the centre of a circle
			- max radius
			- min radius
	*/
	hough(frame, thresh, dir, 180, 210, 40, 30);
	// 4. Save Result Image
	imwrite( "HOUGH.jpg", frame );

	return 0;
}


// ----------------------------------------------------------------------------------------------------
// ---------- SOBEL THINGS --------------------------------------------------
/*
*		our implementation of sobel !!
*/
void sobelDetection( Mat &input, Mat &dx, Mat &dy, Mat &mag, Mat &dir, Mat &sobe){
	// Compute image containing derivative in x direction and y direction
	ddx(input, dx, dy);
	magnitude (input, dx, dy, mag);
	direction (input, dx,dy,dir);	

	normalize(dx, dx, 0, 255, 32, -1);
	normalize(dy, dy, 0, 255, 32, -1);
	imwrite( "ourSobelY.jpg", dy ); // image w deriv in y direction
	imwrite( "ourSobelX.jpg", dx ); // image w deriv in x direction

	// magnitude of gradient
	normalize(mag, mag, 0, 255, 32, -1);
	imwrite("MAGN.jpg", mag);

	normalize(dir,dir,0,255,32,-1);
	imwrite("DIRECTION.jpg", dir);
}


// derivation in x direction
void ddx(cv::Mat &input, cv::Mat &resultX, cv::Mat &resultY ) {
	// need to initialise the output back to sobel
	resultX.create(input.size(), CV_64FC1);
	resultY.create(input.size(), CV_64FC1);

	// initialise kernels
	cv::Mat kernel_X = (Mat_<float>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	cv::Mat kernel_Y = (Mat_<float>(3,3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

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
			resultX.at<double>(i,j) = (double) sumX;			
			resultY.at<double>(i,j) = (double) sumY;
		}
	}	
}


// gradient magnitude
void magnitude (cv::Mat &input, cv::Mat &scaledX, cv::Mat &scaledY, cv::Mat &mag) {
	mag.create(input.size(), CV_64FC1);
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			mag.at<double>(i,j) = sqrt( pow(scaledX.at<double>(i,j), 2) + pow(scaledY.at<double>(i,j), 2) );	
		}
	}
}


// gradient direction
void direction (cv::Mat &input, cv::Mat &scaledX, cv::Mat &scaledY, cv::Mat &dir){
	dir.create(input.size(), input.type());
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			dir.at<uchar>(i,j) = atan( scaledY.at<double>(i,j) / scaledX.at<double>(i,j));
		}
	}
}


// ----------------------------------------------------------------------------------------------------
// ---------- HOUGH THINGS --------------------------------------------------

// -> to get set of pixels with trongest g. magnitude to be considered for circle detection
void thresholding( Mat &mag, Mat &thresh){
	thresh.create(mag.size(), mag.type());
	for (int i = 0; i < mag.rows; i++) {
		for (int j = 0; j < mag.cols; j++ ) {
			if (mag.at<double>(i,j) > 100 ) {
				thresh.at<double>(i,j) = 255;
			}
			else {
				thresh.at<double>(i,j) = 0;
			}
		}
	}

	imwrite("thresh.jpg", thresh);
}


// mag, dir, hspace
// dont need frame
// 190,40,5
// maxR is frame.rows/2

void hough(Mat frame, Mat &mag, Mat &dir, int thresh, int peak, int maxR, int minR){
	vector<cv::Vec3i> detectedDarts;
	if (frame.rows < frame.cols) {
		maxR = frame.rows / 2;
	}
	else maxR = frame.cols / 2;
	
	detectedDarts = houghCircleDetection( mag, dir, thresh, peak, maxR, minR);

	// draw a circle with radius r
	for( int i = 0; i < detectedDarts.size(); i++ ){ 
		Vec3i temp = detectedDarts[i];
		cv::circle(frame, Point(temp[0], temp[1]), temp[2], Scalar( 255, 0, 0 ), 2);
	}

	// display the vector
	std::copy(detectedDarts.begin(), detectedDarts.end(), std::ostream_iterator<cv::Vec3i>(std::cout, " "));
	cout<<endl;
}



// eq of circle: (x-x0)^2 + (y-y0)^2 = r^2
vector<cv::Vec3i> houghCircleDetection( Mat &mag, Mat &dir, int thresh, int peak, int maxR, int minR){
	vector<cv::Vec3i> darts; //struct that holds 3 ints
	int x0pos, x0neg, y0pos, y0neg;
	int rows = mag.rows;
	int cols = mag.cols;

	// What does this value need to be
	// int thresh = 250; // this is a pixel value (ie Ts)

	int ***hspace3D;
	//  Allocate 3D Array
	hspace3D = new int**[cols];
	for(int i = 0; i < cols; i++){
		hspace3D[i] = new int*[rows];
		for(int j = 0; j < rows; j++){
			hspace3D[i][j] = new int[maxR];
      		}
	}

	// step size -> can have step size if we want to make hough calcualtions faster
	// can be a subtask 4 thing?
	// initialize 3D array w 0s
	for(int x=0; x< cols; x++){
		for(int y=0; y< rows; y++){
			for(int r = 0; r < maxR; r++){
				hspace3D[x][y][r] = 0;
			}
		}
	}


	// x and y is the center of a circle
	for (int x=0; x< cols; x++){
		for (int y=0; y< rows; y++){
			  if (mag.at<double>(y,x) == 255 ){
				for (int r = minR; r < maxR; r++){
					// hough based on gradient direction given by dir matrix
					for (int t = dir.at<uchar>(y,x) - 10; t < dir.at<uchar>(y,x) + 10; t++){
						// -ve
						int x0 = x - (int)(r * cos(t));
						int y0 = y - (int)(r * sin(t));
						if (x0 >= 0 && y0 >= 0 && x0 < cols && y0 < rows){
							hspace3D[x0][y0][r] += 1; // plus one vote
						}
						// +ve
						x0 = x + (int)(r * cos(t));
						y0 = y + (int)(r * sin(t));
						if (x0 >= 0 && y0 >= 0 && x0 < cols && y0 < rows){
							hspace3D[x0][y0][r] += 1; // plus one vote
						}
					}
				}
			 }
		}
	}

	// scaling the hough votes so its easier to find peak
	scaling(hspace3D, maxR, cols, rows);

	// 2. in parameter space, any element H(xo, yo, r) > Th
	// represents a circle with radius r located at (xo, yo) in the image
	for (int xo=0; xo< cols; xo++) {
		for (int yo=0; yo< rows; yo++) {
			for (int radius=minR; radius< maxR; radius++) {
				// if it is a peak in the hough space then a circle is detected
				if (hspace3D[xo][yo][radius] > peak) { // this is Th
					// save detected circles in a vector
					darts.push_back(Vec3i(xo, yo, radius));
				}
			}
		}
	}

	cv::Mat hspace2d;
	hspace2d.create(mag.size(), mag.type());

	// displaying hough space image
	for (int x = 0; x < cols; x++){
		for (int y = 0; y < rows; y++){
			for (int r = 0; r < maxR; r++){
				hspace2d.at<double>(y,x) += hspace3D[x][y][r];
			}
        }
    }
	normalize(hspace2d, hspace2d, 0, 255, 32, -1);
	imwrite("hspace.jpg", hspace2d);

	// openCV library func:
	// HoughCircles(mag, hresult, CV_HOUGH_GRADIENT, 1, mag.rows/8, 200, 100, 0, 0 );

	return darts; // returns the vector of all detected darts
}




void scaling( int *** hough, int maxR, int cols, int rows ){
	int max = 0;
	// find the max
	for (int x=0; x< cols; x++){
		for(int y=0; y< rows; y++){
			for (int r=0; r< maxR; r++){
				if (hough[x][y][r] > max){
					max = hough[x][y][r];
				}	
			}
		}
	}
	// scale the thing
	for (int x=0; x< cols; x++){
		for(int y=0; y< rows; y++){
			for (int r=0; r< maxR; r++){
				hough[x][y][r] = ( hough[x][y][r] * 255 ) / max ;	
			}
		}
	}
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


