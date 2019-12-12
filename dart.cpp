/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - dart.cpp
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

/* constants */
const std::string GTFILENAME = "gt_darts_data.txt";

/* struct that contains data OF an IMAge to draw bb */
struct datastruct {
	int x, y, w, h;
};

/** Function Headers */
void detectAndDisplay( string fname, Mat frame );
void getGroundTruthData();
void drawGroundTruth( string fname, Mat frame);
float calcIOU(string fname, int px, int py, int pw, int ph, int col);
float calcTPR(float iou[], int index);
float calcF1score(float iou[], int index, int noOfDetected);


/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;
datastruct gt[16][16];
int no_of_darts[16]={1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 3, 1, 1, 1, 2, 1};

/** @function main */
int main( int argc, const char** argv ){
	// get gt data from file
	getGroundTruthData();

	// 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// draw and display gt boxes
	drawGroundTruth(argv[1], frame);
	std::cout << "ground truth drawn" << std::endl;

	// 3. Detect Faces and Display Result
	detectAndDisplay( argv[1], frame );

	// 4. Save Result Image
	imwrite( "detectedGTDarts.jpg", frame );

	return 0;
}


/** @function detectAndDisplay **/
void detectAndDisplay( string fname, Mat frame ){
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
	for( int i = 0; i < faces.size(); i++ ){
		// print details of faces
        	std::cout << faces[i].x << " " << faces[i].y << " " << faces[i].width << " " << faces[i].height << std::endl;

		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}
}


void getGroundTruthData(){
	cout << "in GT" << endl;

	string line;
	int oldIndex = 99;
	int index;
	string T;
	int col = 0;
	
	ifstream f;
	f.open(GTFILENAME, ios::out);

	// gets first line but we ignore it since its a comment
	std::getline(f, line);
		
	while(getline(f, line)){
		// get a line
		// std::getline(f, line);
		// create a temp thing
		std::stringstream temp(line);

		// extract index from line
		getline(temp, T, ' ');
		index = stoi(T);
    	
		// dealing with images with more than one face/dart
		if (oldIndex != index) {
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

	
		// cout<<"index: " <<index << endl;
		 cout<< gt[index][0].x << " " << gt[index][0].y << " " <<gt[index][0].w << " "<<gt[index][0].w << endl;
	
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
	cout<<"index is: "<< index <<endl;

	// draw rect
	while(gt[index][col].x !=0){
		rectangle(frame, Point(gt[index][col].x, gt[index][col].y), Point(gt[index][col].x + gt[index][col].w, gt[index][col].y + gt[index][col].h), Scalar( 0, 0, 255 ), 2);
		if (gt[index][col].x == 0){
			break;
		}
		 
		col++;
		
	}
}
