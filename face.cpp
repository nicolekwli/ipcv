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

/* struct with ground truth values??? */
struct datastruct {
	int x, y, w, h;
};

// initializing the values
//datatstruct gt_data[4];
//gt_data[0] = 


/** Function Headers */
void detectAndDisplay( Mat frame );
void groundTruth( string fname, Mat frame);
float calcIOU(int l2_x, int l2_y, int r2_x, int r2_y);
void calcF1score();

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{
	// 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// ---> display ground truth boxes
	groundTruth(argv[1], frame);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

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
	for( int i = 0; i < faces.size(); i++ )
	{
		// print details of faces
        std::cout << faces[i].x << " " << faces[i].y << " " << faces[i].width << " " << faces[i].height << std::endl;

		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);

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
// draws the ground truth (red) boxes 
void groundTruth(string fname, Mat frame){
	std::vector<std::tuple<int, int, int, int>> ground_truth_data;
	ground_truth_data.push_back(std::make_tuple(330, 100, 130, 140)); // dart4
	ground_truth_data.push_back(std::make_tuple(0, 0, 0, 0)); //dart5
	ground_truth_data.push_back(std::make_tuple(0, 0, 0, 0)); //dart13
	ground_truth_data.push_back(std::make_tuple(0, 0, 0, 0)); //dart14
	ground_truth_data.push_back(std::make_tuple(0, 0, 0, 0)); //dart15

	//std::cout << "getting data" << std::get<0>(ground_truth_data[0]) << std::endl;

	if (fname == "dart4.jpg"){
		rectangle(frame, Point(330, 100), Point(460, 240), Scalar( 0, 0, 255 ), 2);
		// rectangle(frame, Point(std::get<0>(ground_truth_data[0]), std::get<0>(ground_truth_data[1])), Point(460, 240), Scalar( 0, 0, 255 ), 2);
	}
	// else if (fname == "dart5.jpg"){
	// 	rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 0, 255 ), 2);
	// }
	// else if (fname == "dart13.jpg"){
	// 	rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 0, 255 ), 2);
	// }
	// else if (fname == "dart14.jpg"){
	// 	rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 0, 255 ), 2);
	// }
	// else if (fname == "dart15.jpg"){
	// 	rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 0, 255 ), 2);
	//}
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
	
}


void calcF1score(){

}