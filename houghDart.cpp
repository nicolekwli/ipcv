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
#include <fstream>

using namespace std;
using namespace cv;


/** Function Headers */
// THESE ARE FROM FACE.CPP
void getGroundTruthData();
void drawGroundTruth( string fname, Mat frame);
float calcIOU(string fname, int px, int py, int pw, int ph, int col);
float calcTPR(float iou[], int index);
float calcF1score(float iou[], int index, int noOfDetected);
void doCalc( string fname, Mat frame, std::vector<Rect> dart );

vector<Rect> violaJones(string fname, Mat frame);

void sobelDetection( Mat &input, Mat &dx, Mat &dy, Mat &mag, Mat &dir);
void ddx(cv::Mat &input, cv::Mat &resultX, cv::Mat &resultY);
void magnitude(cv::Mat &input, cv::Mat &ddx, cv::Mat &ddy, cv::Mat &magnitude);
void direction(cv::Mat &input, cv::Mat &ddx, cv::Mat &ddy, cv::Mat &direction);
void thresholding(Mat &input, Mat &thresholded);

vector<Rect> hough(Mat frame, Mat &mag, Mat &dir, int thresh, int peak, int maxR, int minR);
vector<cv::Vec3i> houghCircleDetection( Mat &mag, Mat &dir, int thresh, int peak, int maxR, int minR);
void scaling(int *** hough, int maxR, int cols, int rows);

void voilaAndHough(string name, Mat frame, Mat frame_gray );
vector<Rect> combineResults(vector<Rect> input);
/**************************/


/* struct that contains data OF an IMAge to draw bb */
struct datastruct {
	int x, y, w, h;
};


/** Global variables */
const std::string GTFILENAME = "gt_darts_data.txt";
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;
datastruct gt[16][16];
int no_of_darts[16] = {1,1,1,1,1,1,1,1,2,1,3,1,1,1,2,1};


/** @function main */
int main( int argc, const char** argv ){
	// Read Input Image
	cv::Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	cv::Mat frame_gray = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

	getGroundTruthData();
	drawGroundTruth(argv[1], frame);
	cout << " -> ground truth read and drawn" << endl;
	
	voilaAndHough(argv[1], frame, frame_gray);

	return 0;
}

// ----------------------------------------------------------------------------------------------------
// ---------- Ground truth and friends  ---------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------
void getGroundTruthData(){
	string line;
	int oldIndex = 99;
	int index;
	string T;
	int col = 0;

	ifstream f;
	f.open(GTFILENAME, ios::out);
	
	// gets first line but we ignore it since its a comment
	std::getline(f, line);
	while(std::getline(f, line)){
		// create a temp thing
		std::stringstream temp(line);

		// extract first value which is the index from line
		getline(temp, T, ' ');
		index = stoi(T);
    	
		if (oldIndex != index){
			col = 0;
		}

		// extracting each value for x y w h as integer
		getline(temp, T, ' ');
		gt[index][col].x = stoi(T);
		getline(temp, T, ' ');
		gt[index][col].y = stoi(T);
		getline(temp, T, ' ');
		gt[index][col].w = stoi(T);
		getline(temp, T, ' ');
		gt[index][col].h = stoi(T);

		if (index == 15){
			break;
		}
		col++;
		oldIndex = index;
	}
	f.close();
}

/* draws the ground truth (red) boxes */
void drawGroundTruth(string fname, Mat frame){
	int index = 0;
	int col = 0;
	// get image number ie index as in int
	index = fname[4]-48;
	char dot = '.';
	if (fname[5] != dot){
		index = stoi(to_string(index) + to_string(fname[5] - 48));
	}
	// draw rect
	while(gt[index][col].x != 0){
		rectangle(frame, Point(gt[index][col].x, gt[index][col].y), Point(gt[index][col].x + gt[index][col].w, gt[index][col].y + gt[index][col].h), Scalar( 0, 0, 255 ), 2);
		if (gt[index][col].x == 0){
			break;
		}
		col++;
	}
}


void doCalc( string fname, Mat frame, std::vector<Rect> faces ){
	int index = fname[4]-48;
	char dot = '.';
	if (fname[5] != dot){
		index = stoi(to_string(index) + to_string(fname[5] - 48));
	}
	// need variable to keep track of which one has been included
	// lets use a new array of with size of the number of actualy faces
	int facesMatched[no_of_darts[index]]; 
	
	// NEED A LIST OF IOU TO PASS INTO TPR
	float IOUs[no_of_darts[index]];

	cout << "INDEX: " << index << std::endl;
	cout << "NO OF DARTS: " << no_of_darts[index] << endl;
	// for each ground truth face we perform IOU with all detected faces
	for (int j = 0; j < no_of_darts[index]; j++){
		float IOU = 0;
		int IOUIndex;
		float tempIOU;

		// for each detected face store the largest one
		for (int k = 0; k < faces.size(); k++){
			// perform IOU and compare to previous and store the larger one	
			tempIOU = calcIOU(fname, faces[k].x, faces[k].y, faces[k].width, faces[k].height, j);
			// cout << "x: "<< faces[k].x << endl;
			// cout << "y: "<<  faces[k].y<< endl; 
			// cout << "width: " << faces[k].width << endl;
			// cout << "height: " << faces[k].height << endl;
			if (IOU < tempIOU){
				if (j == 0){
					IOU = tempIOU;
				}
				// replace and keep larger one
				// store the actual order of detected faces in a new array
				
				// if the K is not in the facesMatched array, then store otherwise its no good
				for (int l = 0; l < j; l++) {
					if (l == 0){
						IOU = tempIOU;
					}
					else if (k != facesMatched[l]){
						IOU = tempIOU;	
						IOUIndex = k;
					}
				}
			} 		
		}
		// store final one in facesMatched
		facesMatched[j] = IOUIndex;
		IOUs[j] = IOU;
		// display the best IOU
		std::cout << "iou: " << IOU << std::endl;
	}
	std::cout << "TPR: " << std::endl;
	float tpr = calcTPR(IOUs, index);
	std::cout << tpr << std::endl;

	// F1
	// no of detected should increment whenever one we dont include
	int noOfDetected = faces.size();
	cout << "detected: " << noOfDetected << endl;
	std::cout << "F1: " << std::endl; 
	std::cout << calcF1score(IOUs, index, noOfDetected) << std::endl;

}

float calcIOU(string fname, int px, int py, int pw, int ph, int col){
	// cout << "COLE: "<< col << endl;
	// - get index & declare variables
	int index = fname[4]-48;
	char dot = '.';
	if (fname[5] != dot){
		index = stoi(to_string(index) + to_string(fname[5] - 48));
	}
	float iou = 0.0, intersect_area = 0.0;
	float union_area;
	int g_area, p_area = 0; 

	int gx = gt[index][col].x;
	int gy = gt[index][col].y;
	int gw = gt[index][col].w;
	int gh = gt[index][col].h;

	// - area of each rect is width x height
	p_area = pw * ph;
	// g_area = gt[index][col].w * gt[index][col].h;
	g_area = gw * gh;

	// - get points of the intersecting rectangle i.e l2 and r1 resp.
	// top left, bottom right
	int ix1, iy1, ix2, iy2;
	ix1 = min(gx + gw, px + pw);
	ix2 = max(gx, px);
	iy1 = min(gy + gh, py + ph);
	iy2 = max(gy, py);

	// - calc the area of intersection
	if (((ix1 - ix2) > 0) && ((iy1 - iy2) > 0)){
		intersect_area = abs(ix1 - ix2) * abs(iy1 - iy2);		
	}
	else return 0;
	// determine union area
	union_area = (g_area + p_area) - intersect_area;
	
	// iou of the box
	iou = abs(intersect_area) / abs(union_area);
	return iou;
}

float calcTPR(float iou[], int index){
	int TP = 0;
	// for each IOU value
	for (int i = 0 ; i < no_of_darts[index]; i++){
		if (iou[i] >= 0.5 ){
			// then it is a true positivei
			TP = TP + 1;
		}
	}
	float tpr = (float) TP / no_of_darts[index];
	return tpr;
}

float calcF1score(float iou[], int index, int noOfDetected){
	int TP = 0;
	int FN = 0;
	int FP = noOfDetected - no_of_darts[index];
	if (FP < 0){
		FP = 0;
	}
	for (int i = 0 ; i < no_of_darts[index]; i++){
	cout << "IOU VALUE: "<< iou[i] << endl;
		if (iou[i] >= 0.5 ){
			// then it is a true positive
			TP = TP + 1;
		}
		if (iou[i] == 0) {
			FN = FN + 1;
		}	
	}
	// cout << "fp: " << FP << endl;
	// cout << "tp: " << TP << endl;
	// cout << "fn: " << FN << endl;
	// ACTUAL F1	
	float precision = (float) TP / (TP + FP);
	float recall = (float) TP / (TP + FN);
	// cout<< "precision: " << precision << endl;
	// cout << "recall: " << recall << endl;
	return 2 * ((precision * recall)/(precision + recall));
}


// ----------------------------------------------------------------------------------------------------
// ---------- SOBEL EDGE DETECTION  -------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------
void sobelDetection( Mat &input, Mat &dx, Mat &dy, Mat &mag, Mat &dir){
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
// ---------- HOUGH TRANSFORM -------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------
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
// maxR is frame.rows/2
vector<Rect> hough(Mat frame, Mat &mag, Mat &dir, int thresh, int peak, int maxR, int minR){
	cout<< " -> ... hough in progress " <<endl;
	vector<cv::Vec3i> detectedDarts;
	vector<Rect> detectedDartBoxes;
	if (frame.rows < frame.cols) {
		maxR = frame.rows / 2;
	}
	else maxR = frame.cols / 2;
	
	detectedDarts = houghCircleDetection( mag, dir, thresh, peak, maxR, minR);

	// draw a circle with radius r
	// also create vector of all detected rects
	for( int i = 0; i < detectedDarts.size(); i++ ){ 
		Vec3i temp = detectedDarts[i];
		// cv::circle(frame, Point(temp[0], temp[1]), temp[2], Scalar( 255, 0, 0 ), 2);

		detectedDartBoxes.push_back(Rect(temp[0]-temp[2], temp[1]-temp[2], temp[2]*2, temp[2]*2));
		// cv::rectangle(frame, Point(temp[0]-temp[2], temp[1]-temp[2]), Point(temp[0]+temp[2], temp[1] + temp[2]), Scalar( 0, 255, 0 ), 2);
	}

	std::cout << " -> no of darts detected by Hough: " << detectedDartBoxes.size() << std::endl;

	return detectedDartBoxes;
}


// describe me
vector<cv::Vec3i> houghCircleDetection( Mat &mag, Mat &dir, int thresh, int peak, int maxR, int minR){
	vector<cv::Vec3i> darts; //struct that holds 3 ints
	int x0pos, x0neg, y0pos, y0neg;
	int rows = mag.rows;
	int cols = mag.cols;

	int ***hspace3D;
	//  Allocate 3D Array
	hspace3D = new int**[cols];
	for(int i = 0; i < cols; i++){
		hspace3D[i] = new int*[rows];
		for(int j = 0; j < rows; j++){
			hspace3D[i][j] = new int[maxR];
		}
	}

	// initialize 3D array w 0s
	for(int x=0; x< cols; x++){
		for(int y=0; y< rows; y++){
			for(int r = 0; r < maxR; r++){
				hspace3D[x][y][r] = 0;
			}
		}
	}
		
		cout<< " -> ... hough in progress " <<endl;

	// the actual calculation & casting votes
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

		cout<< " -> ... hough in progress " <<endl;

	// scaling the hough votes so its easier to find peak
	scaling(hspace3D, maxR, cols, rows);

	// using peak threshold given to get circles detected
	for (int xo=0; xo< cols; xo++) {
		for (int yo=0; yo< rows; yo++) {
			for (int radius=minR; radius< maxR; radius++) {
				// if it is a peak in the hough space then a circle is detected
				if (hspace3D[xo][yo][radius] > peak) {
					// save detected circles in a vector
					darts.push_back(Vec3i(xo, yo, radius));
				}
			}
		}
	}

	cv::Mat hspace2d;
	hspace2d.create(mag.size(), mag.type());

	// displaying hough space image => convert to 2D
	for (int x = 0; x < cols; x++){
		for (int y = 0; y < rows; y++){
			for (int r = 0; r < maxR; r++){
				hspace2d.at<double>(y,x) += hspace3D[x][y][r];
			}
        }
    }

	normalize(hspace2d, hspace2d, 0, 255, 32, -1);
	imwrite("hspace.jpg", hspace2d);

	return darts; // returns the vector of all detected darts
}

// scaling the votes to be out of 255
void scaling( int *** hough, int maxR, int cols, int rows ){
	cout<< " -> ... hough in progress " <<endl;
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



// ----------------------------------------------------------------------------------------------------
// ---------- INTEGRATION W VIOLA JONES ---------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------
vector<Rect> violaJones( string fname, Mat frame ) {
	std::vector<Rect> dart;
	Mat frame_gray, result;

	// Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); exit(0); };

	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, dart, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
	
	std::cout << " -> no of darts detected by viola-jones: " << dart.size() << std::endl;

	/*for( int i = 0; i < dart.size(); i++ ){
		rectangle(frame, Point(dart[i].x, dart[i].y), Point(dart[i].x + dart[i].width, dart[i].y + dart[i].height), Scalar( 165, 192, 203 ), 2);
	}*/

	// ANALYSIS CALC
	// doCalc(fname, frame, dart);

	return dart;
}



/*
* 		Viola Jones + Hough transform 
*/
void voilaAndHough(string name, Mat frame, Mat frame_gray ){
	cv::Mat dx, dy, mag, dir, thresh;
	vector<Rect> VJresult, Hresult, VJaveraged, Haveraged;

	// do viola
	VJresult = violaJones(name, frame);
	cout << " -> viola jones done" << endl;

	// do hough
	sobelDetection(frame_gray, dx, dy, mag, dir);
	thresholding(mag, thresh);
	Hresult = hough(frame, thresh, dir, 180, 160, 40, 45);
	cout << " -> hough detection done" << endl;

	// combine the results -----------------------------------------------
	vector<Rect> reducedResult;
	vector<Rect> intersections;


	// reduce the number of results by combining those that intersect
	 Hresult = combineResults(Hresult);
	/*for (int j = 0; j< Hresult.size(); j++){
		rectangle(frame, Point(Hresult[j].x, Hresult[j].y), Point(Hresult[j].x + Hresult[j].width, Hresult[j].y + Hresult[j].height), Scalar(255,192,203),2);
	}*/	
	//	VJresult = combineResults(VJresult);
	/*for( int i = 0; i < VJresult .size(); i++ ){
		rectangle(frame, Point(VJresult [i].x, VJresult [i].y), Point(VJresult [i].x + VJresult [i].width, VJresult [i].y + VJresult [i].height), Scalar( 255, 192, 203 ), 2);
	}*/
	// We assume that both detection should have detected part of the dartboard
	//for each hough detected rect compare with each viola detected rect
	for (int i = 0; i < Hresult.size(); i++){
		vector<Rect> Hintersect;

		for (int j = 0; j < VJresult.size(); j++) {
			// check if they intersect
			Rect check = Hresult[i] & VJresult[j];
			// think this hsould be the threshold
			if (check.area() > (0) ){
				// they intersect
				Hintersect.push_back(check);
			}
			else {
				// they dont intersect so erase the jth rectangle element
				// VJresult.erase(VJresult.begin()+j);
			}
		}

		if (Hintersect.empty()){
			// if no interesections b/w H and VJ then delete that H rectangle
			// based on assumption we decide that not a bit of the dartboard is detected.
			Hresult.erase(Hresult.begin()+i);
		}
		else {
			intersections.insert(intersections.end(), Hintersect.begin(), Hintersect.end());
		}
	}

	cout<<"intersections " << intersections.size() <<endl;
	// ----------------------------------------------------------------------------

	// filtering boxes that overlap in the same place
	groupRectangles(intersections, 0, 0.2);
	cout<< " -> filtering done " <<endl;

	// draw reduced result rectangles
	std::cout << " -> no of darts detected by viola-jones AND hough: " << intersections.size() << std::endl;
	for( int i = 0; i < intersections.size(); i++ ){
		rectangle(frame, Point(intersections[i].x, intersections[i].y), Point(intersections[i].x+intersections[i].width, intersections[i].y+intersections[i].height), Scalar( 255, 255, 0 ), 2);
	}

	// Save Result Image
	imwrite( "VJH.jpg", frame );

	// ANALYSIS
	doCalc( name, frame, intersections );
}


// used to sort vector of Rects in ASC order
struct compareAsc {
    bool operator () (const Rect &a, const Rect &b) {
		return a.x < b.x;
    }
};

// this function detects overlapping detected boxes and combines them, reducing the number of boxes.
vector<Rect> combineResults(vector<Rect> input){
	vector<Rect> newInput;
	vector<Rect> inputCopy = input;
	for (int i = 0; i < input.size(); i++) {
		// if the intersect, store
		vector<Rect> intersect;

		for (int j = 0; j < inputCopy.size(); j++) {
			if (i != j){
			// check if they intersect
				Rect check = input[i] & inputCopy[j];
				// if its bigger than 1/4 of the area 
				if (check.area() > 0){
					// they intersect
					intersect.push_back(inputCopy[j]);
					inputCopy.erase(inputCopy.begin()+j);
				}
			}
		}
		// if intersect is empty, keep the ith one
		if (intersect.empty()){	
			intersect.push_back(input[i]);
		}
		// if its not empty, add ith to intersect list and conbine all intersect
		else {
			intersect.push_back(input[i]);
			// included this here coz idk
			newInput.insert(newInput.end(), intersect.begin(), intersect.end());
		}
	// combine all thost that need to be at the end of each i-loop
	// and store
	}
	groupRectangles(newInput, 1, 1.6);
	cout << "new detected boxes: " << newInput.size() << endl;
	return newInput;
}