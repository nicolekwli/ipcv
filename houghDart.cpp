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
void scaling(Mat &dx, Mat &dy, Mat &scaledX, Mat &scaledY);

void detectAndDisplay( Mat frame );
void ddx(cv::Mat &input, cv::Mat &resultX, cv::Mat &resultY);
void magnitude(cv::Mat &input, cv::Mat &ddx, cv::Mat &ddy, cv::Mat &magnitude);
void direction(cv::Mat &input, cv::Mat &ddx, cv::Mat &ddy, cv::Mat &direction);
void thresholding(Mat &input, Mat &thresholded);
void hough(Mat frame, Mat &mag, Mat &dir, int peak, int maxR, int minR);
vector<cv::Vec3i> houghCircleDetection( Mat &mag, Mat &dir, int peak, int maxR, int minR);


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

	// declaring all the matrices here
	cv::Mat dx, dy, mag, dir, thresh, gblur, sobe, so;

	//sobelOpenCV(frame, so);
	sobelDetection(frame, dx, dy, mag, dir, sobe);

	// thresholding the gradient magnitude image
	thresholding(mag, thresh);

	// hough
	hough(frame, mag, dir, 190, 40, 5);

	// 4. Save Result Image
	imwrite( "oursobel.jpg", mag );

	return 0;
}


/*
*	 this is sobel's using openCVs library
*/
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
  	// imwrite( "sobelX.jpg", abs_grad_x );

  /// Gradient Y
  Sobel( input, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y );
	// imwrite( "sobelY.jpg", abs_grad_y );

  /// Total Gradient (approximate)
  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobel );
}


/*
*		our implementation of sobel !!
*/
void sobelDetection( Mat &input, Mat &dx, Mat &dy, Mat &mag, Mat &dir, Mat &sobe){

	Mat abs_grad_x, abs_grad_y;
	sobe.create(input.size(), input.type());
	mag.create(input.size(), input.type());

	//GaussianBlur( input, input, Size(3,3), 0, 0, BORDER_DEFAULT ); -> makes it only slightly better

	// Compute image containing derivative in x direction and y direction
	ddx(input, dx, dy);
	// using openCV's scaling function
			// convertScaleAbs( dx, abs_grad_x );
			// convertScaleAbs( dy, abs_grad_y );

	// using our own scaling function
	cv::Mat scaledX, scaledY;
	scaling(dx, dy, scaledX, scaledY);

  	addWeighted(scaledX, 0.5, scaledY, 0.5, 0, sobe ); // need this to combine result x and result y
	imwrite( "ourSobelY.jpg", scaledY ); // image w deriv in y direction
	imwrite( "ourSobelX.jpg", scaledX ); // image w deriv in x direction

	// magnitude of gradient
	magnitude(sobe, scaledX, scaledY, mag);

	// direction of gradient
	// yeah this may or may not work? i know its meant to look bad but i cant tell if its meant to look THIS bad (lol)
	direction(sobe, scaledX, scaledY, dir);
	imwrite( "dir.jpg", dir );

			// convert sobel to image format
			// may not need this if we're already scaling the image
			// this was only needed when we used 32FC1 since that is not an image format
			// whereas CV_8UC1 is an image format
			// sobe.convertTo(sobe, CV_8UC1);
}


// derivation in x direction
void ddx(cv::Mat &input, cv::Mat &resultX, cv::Mat &resultY ) {
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



void scaling(Mat &dx, Mat &dy, Mat &scaledX, Mat &scaledY){
	scaledX.create(dx.size(), dx.type()); 
	scaledY.create(dy.size(), dy.type()); 

	for (int i = 0; i < dx.rows; i++) {
		for (int j = 0; j < dx.cols; j++) {
			/*
			so we have(HAD?idk where they went) -ve and +ve values
			need to scale it from 0 to 255 
			all -ve values are values <128 and +ve values are >128
			*/

			scaledX.at<uchar>(i,j) = 128 + dx.at<uchar>(i,j); 
			scaledY.at<uchar>(i,j) = 128 + dy.at<uchar>(i,j);

			// somehow i accidentally fixed things?
			// the above logic is only meant for the case with -ve and +ve
		}
	}
}



// magnitude of gradient for sobel
// pythagras theorem!!!!!!!!!!!!!!! SSS HH OOO KK
// how big is the edge at this direction
// scale it??
void magnitude (cv::Mat &input, cv::Mat &scaledX, cv::Mat &scaledY, cv::Mat &mag) {
	mag.create(input.size(), input.type());
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			mag.at<uchar>(i,j) = sqrt( pow(scaledX.at<uchar>(i,j), 2) + pow(scaledY.at<uchar>(i,j), 2) );	
		}
	}
}


// the image of this LOOKS BAD dont be afraid
void direction (cv::Mat &input, cv::Mat &scaledX, cv::Mat &scaledY, cv::Mat &dir){
	dir.create(input.size(), input.type());
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			dir.at<uchar>(i,j) = atan( scaledY.at<float>(i,j) / scaledX.at<float>(i,j));
		}
	}

	//cout << "KY is = " << endl << " " << dir << endl << endl;

}



// technically, code-wise, this works
// but image doesnt look great? need a better thresh value
// -> to get set of pixels with trongest g. magnitude to be considered for circle detection
void thresholding( Mat &mag, Mat &thresh){
	thresh.create(mag.size(), mag.type());
	for (int i = 0; i < mag.rows; i++) {
		for (int j = 0; j < mag.cols; j++ ) {
			if (mag.at<uchar>(i,j) > 190 ) {
				thresh.at<uchar>(i,j) = 255;
			}
			else {
				thresh.at<uchar>(i,j) = 0;
			}
		}
	}

	imwrite("thresh.jpg", thresh);
}



void hough(Mat frame, Mat &mag, Mat &dir, int peak, int maxR, int minR){
	vector<cv::Vec3i> detectedDarts;
	// houghLineDetection();
	detectedDarts = houghCircleDetection( mag, dir, peak, maxR, minR);

	// draw a circle with radius r
	for( int i = 0; i < detectedDarts.size(); i++ ){ 
		Vec3i temp = detectedDarts[i];
		cv::circle(frame, Point(temp[0], temp[1]), temp[2], Scalar( 0, 0, 255 ), 2);
	}

	// view the vector
	std::copy(detectedDarts.begin(), detectedDarts.end(), std::ostream_iterator<cv::Vec3i>(std::cout, " "));
}



// eq of circle: (x-x0)^2 + (y-y0)^2 = r^2
vector<cv::Vec3i> houghCircleDetection( Mat &mag, Mat &dir, int peak, int maxR, int minR){
	// should this be an array or vector
	int rows = mag.rows, cols = mag.cols;
	int hspace[rows][cols][maxR];
	int thresh = 250; // this is a pixel value (ie Ts)
	vector<cv::Vec3i> darts; //struct that holds 3 ints

	// step size -> can have step size if we want to make hough calcualtions faster
	// can be a subtask 4 thing?

	// initialize w 0s
	for(int x=0; x< rows; x++){
		for(int y=0; y< cols; y++){
			for(int r = minR; r < maxR; r++){
				hspace[x][y][r] = 0;
			}
		}
	}

	int x0pos, x0neg, y0pos, y0neg;
		// 1. for any pixel satisfying |Mag(x,y)| > Ts, increment all elements satisying the following:
		//		for all r, xo = x +- rcos(dir(x,y))
		//				   yo = y +- rsin(dir(x,y))
		//		H(xo, yo, r) = H(xo, yo, r) + 1 ----> gets one vote!

	// x and y is the center of a circle
	for (int x=0; x< rows; x++){
		for (int y=0; y< cols; y++){

			// we do this since we only care about edges
			if (mag.at<double>(x,y) > thresh ){

				// for all radii from min to max possible r
				for (int r = minR; r < maxR; r++){
					// hough based on gradient direction given by dir matrix
					int x0, y0;
					// -ve
					x0 = x - (int)(r * cos(dir.at<double>(y, x)));
					y0 = y - (int)(r * sin(dir.at<double>(y, x)));
					if (x0 >= 0 && y0 >= 0 && x0 < rows && y0 < cols){
						hspace[x0][y0][r] += 1; // plus one vote
					}
					// +ve
					x0 = x + (int)(r * cos(dir.at<double>(y, x)));
					y0 = y + (int)(r * sin(dir.at<double>(y, x)));
					if (x0 >= 0 && y0 >= 0 && x0 < rows && y0 < cols){
						hspace[x0][y0][r] += 1; // plus one vote
					}
				}
			}

		}
	}

		// 2. in parameter space, any element H(xo, yo, r) > Th
		//		represents a circle with radius r located at (xo, yo) in the image

	for (int xo=0; xo< rows; xo++) {
		for (int yo=0; yo< cols; yo++) {
			for (int radius=0; radius< maxR; radius++) {
				// if it is a peak in the hough space then a circle is detected
				if (hspace[xo][yo][radius] > peak) { // this is Th
					// save detected circles in a vector
                    darts.push_back(Vec3i(xo, yo, radius));
				}
			}
		}
	}

	cv::Mat hspace2d;
	hspace2d.create(Size(mag.rows, mag.cols), mag.type()); // is prob wrong
	// displaying hough space image
	for (int r = minR; r <= maxR; r++){
        for (int x = 0; x < mag.rows; x++){
            for (int y = 0; y < mag.cols; y++){
                hspace2d.at<uchar>(y,x) += hspace[x][y][r];
			}
        }
    }

	imwrite("hspace.jpg", hspace2d);

	// openCV library func:
	// HoughCircles(mag, hresult, CV_HOUGH_GRADIENT, 1, mag.rows/8, 200, 100, 0, 0 );

	return darts; // returns the vector of all detected darts
}



void houghLineDetection(){
	int slope, yint;
	int hough[23][23];
	// mat vs array of ints vs vector
	// make this matrix 0s

	// try different values of theta since we have x and y
	// rho is unknown

	// put values of rho and theta into said matrix

	// the perpendicular from the origin to any point on that line will always be same
	// and thus have the same rho and theta --> so they get more votes

	// general algorithm
	// for all x
	// 	for all y
	// 			if edge point at (x,y)
	// 				for all theta
	// 					rho = x*cos theta + y*sin theta
	// 					H<theta, rho> += 1
	// 				end
	// 			end
	// 	end
	// end
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


