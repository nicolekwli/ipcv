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
#include <fstream>

using namespace std;
using namespace cv;

/* constants */
const std::string GTFILENAME = "gt_faces_data.txt";

/* struct that contains data OF an IMAge to draw bb */
struct datastruct {
	int x, y, w, h;
};

/** Function Headers */
void detectAndDisplay( Mat frame );
void getGroundTruthData();
void drawGroundTruth( string fname, Mat frame);
float calcIOU(int l2_x, int l2_y, int r2_x, int r2_y);
void calcF1score();

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;
datastruct gt[15][15];


/** @function main */
int main( int argc, const char** argv ){
	// get gt data from file
	getGroundTruthData();

	// 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// ---> display ground truth boxes
	drawGroundTruth(argv[1], frame);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

	// 4. Save Result Image
	imwrite( "drawn.jpg", frame );

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame ){
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and 
	//normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

	// 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

	// 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ ){
		// print details of faces
        std::cout << faces[i].x << " " << faces[i].y << " " << faces[i].width << " " << faces[i].height << std::endl;

		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);

		// use the struct to store these in another global array?
		int l2_x, l2_y, r2_x, r2_y;
		l2_x = faces[i].x;
		l2_y = faces[i].y;
		r2_x = faces[i].x + faces[i].width;
		r2_y = faces[i].y + faces[i].height;

		// pass fname here?
		calcIOU(l2_x, l2_y, r2_x, r2_y);
	}
}

// ----- SUBTASK 1 ----------------------------------------------------------------

void getGroundTruthData(){
	string line;
	int index;
	ifstream f;
	f.open(GTFILENAME, ios::out);
	string T;
	
	// gets first line but we ignore it since its a comment
	std::getline(f, line);
		
	while(!f.eof()){
		// get a line
		std::getline(f, line);
		// create a temp thing
		std::stringstream temp(line);

		// extract index from line
		getline(temp, T, ' ');
		index = stoi(T);
    	
		// dealing with images with more than one face/dart
		// is added as a new column
		int col = 0;
		if(gt[index][col].x != 0){
			col++;
		}
		else col = 0;

		// extracting each value for x y w h as integer
		getline(temp, T, ' ');
		gt[index][col].x = stoi(T);
		getline(temp, T, ' ');
		gt[index][col].y = stoi(T);
		getline(temp, T, ' ');
		gt[index][col].w = stoi(T);
		getline(temp, T, ' ');
		gt[index][col].h = stoi(T);
		
		// cout<<"index: " <<index << endl;
		// cout<< gt[index][0].x << " " << gt[index][0].y << " " <<gt[index][0].w << " "<<gt[index][0].w << endl;
	}

	f.close();
}


/* draws the ground truth (red) boxes */
void drawGroundTruth(string fname, Mat frame){
	cout<<"hereeeeeeeeeeeee"<< endl;
	int index = 0;
	int col = 0;
	// get image number ie index
	index = fname[4]-48;
	cout<<"index is: "<< index <<endl;

	// draw rect
	while(gt[index][col].x !=0){
		rectangle(frame, Point(gt[index][col].x, gt[index][col].y), Point(gt[index][col].x + gt[index][col].w, gt[index][col].y + gt[index][col].h), Scalar( 0, 0, 255 ), 2);
		if (gt[index][col].x == 0){
			break;
		}
		else 
			col++;
	}

	// if (fname == "dart4.jpg"){
	// 	rectangle(frame, Point(330, 100), Point(485, 260), Scalar( 0, 0, 255 ), 2);
	// }
	// else if (fname == "dart5.jpg"){
	// 	// x1 y1 x2 y2
	// 	int coords[11][11][11][11];
	// 	for (int i=0; i<11; i++){
	// 		rectangle(frame, Point(250, 164), Point(310, 252), Scalar( 0, 0, 255 ), 2);
	// 	}
	// }
	// else if (fname == "dart6.jpg"){
	// 	rectangle(frame, Point(285, 115), Point(324, 157), Scalar( 0, 0, 255 ), 2);
	// }
	// else if (fname == "dart7.jpg"){
	// 	rectangle(frame, Point(337, 189), Point(428, 286), Scalar( 0, 0, 255 ), 2);
	// }
	// else if (fname == "dart9.jpg"){
	// 	rectangle(frame, Point(81, 200), Point(190, 330), Scalar( 0, 0, 255 ), 2);
	// }
	// else if (fname == "dart11.jpg"){
	// 	rectangle(frame, Point(328, 76), Point(385, 150), Scalar( 0, 0, 255 ), 2);
	// }
	// else if (fname == "dart13.jpg"){
	// 	rectangle(frame, Point(420, 123), Point(544, 255), Scalar( 0, 0, 255 ), 2);
	// }
	// else if (fname == "dart14.jpg"){
	// 	rectangle(frame, Point(461, 210), Point(550, 320), Scalar( 0, 0, 255 ), 2);
	// 	rectangle(frame, Point(725, 185), Point(826, 295), Scalar( 0, 0, 255 ), 2);
	// }
	// else if (fname == "dart15.jpg"){
	// 	rectangle(frame, Point(65, 123), Point(140, 212), Scalar( 0, 0, 255 ), 2);
	// 	rectangle(frame, Point(369, 110), Point(455, 205), Scalar( 0, 0, 255 ), 2);
	// 	rectangle(frame, Point(542, 121), Point(642, 221), Scalar( 0, 0, 255 ), 2);
	// }
}

/*
l1----------
 |			|
 |	 l2-----|-------
  ---|-----r1		|
	 |				|
	 |				|
	  -------------r2
*/

float calcIOU(int l2_x, int l2_y, int r2_x, int r2_y){
	float iou;
	int total, gt_area, predict_area = 0.0; 
	int len, bred;
	float intersection;

	// dart4 coordinates:
	int l1_x, l1_y, r1_x, r1_y;
	l1_x = 330;
	l1_y = 100;
	r1_x = 460;
	r1_y = 240;

	// area for each bounding box
		// width x height ???
	gt_area = (r1_x - l1_x) * (r1_y - l1_y);
	predict_area = (r2_x - l2_x) * (r2_y - l2_y);

	// determine intersection area
	intersection = (min(r1_x, r2_x) - max(l1_x, l2_x)) *  
                (min(r1_y, r2_y) - max(l1_y, l2_y)); 		

	// determine union area
	total = ((gt_area + predict_area) - intersection);
	
	// iou of the box
	iou = abs(intersection) / abs(total);

	// return iou
	std::cout << "iou: " << iou << std::endl;
	return iou;
}

void calcTPR(){
	// if iou >= 0.5 then TP
	// True P Rate is the fraction ofsuccessfully detected faces out of all valid faces in an image
}


void calcF1score(){
	// formula: 2 * ([precision * recall]/[precision + recall])

	// precision = TP / (TP + FP)

	// recall = TP / (TP + FN)

	// we need:
	// TP(from iou)
	// TN(no of detected - no of actual faces) -> faces detected though they dont exist
	// FP(no detected - no of actual) -> no faces but some are detected 
	// FN(no of faces - no detected) -> there are faces but not detected

/*

positive class: % detected by the detector or face
negative class: % not detected by detector or no face

*/

}

/*
250 164 57 57
513 177 55 55
641 184 59 59
191 214 65 65
425 231 68 68
290 242 63 63
58 249 64 64
554 244 69 69
673 246 64 64
695 599 59 59
60 135 63 63
377 190 57 57
384 400 81 81
528 482 170 170
*/