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
// const std::string GTFILENAME = "gt_darts_data.txt";

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
void doCalc( string fname, Mat frame, std::vector<Rect> faces );

/** Global variables */
String cascade_name = "frontalface.xml";
// String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;
datastruct gt[16][16];
int no_of_faces[16] = {0, 0, 0, 0, 1, 11, 1, 1, 0, 1, 0, 1, 0, 1, 2, 0};
// int no_of_face[16] = {1,1,1,1,1,1,1,1,2,1,3,1,1,1,2,1};


/** @function main */
int main( int argc, const char** argv ){
	// get gt data from file
	getGroundTruthData();
	std::cout << " -> data read" << std::endl;

	// 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// ---> display ground truth boxes
	drawGroundTruth(argv[1], frame);
	std::cout << " -> ground truth drawn" << std::endl;

	// 3. Detect Faces and Display Result
	detectAndDisplay( argv[1], frame );

	// 4. Save Result Image
	imwrite( "faceGT.jpg", frame );

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( string fname, Mat frame ){
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and 
	//normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

	// 3. Print number of Faces found
	std::cout << "no. faces detected: " << faces.size() << std::endl;

	// 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ ){
		// print details of faces
		// std::cout << "data of detected face #"<< i+1 << " is: " ;
		// cout<< faces[i].x << " " << faces[i].y << " " << faces[i].width << " " << faces[i].height << std::endl;
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

	doCalc( fname, frame, faces);

}

// ----- SUBTASK 1 ----------------------------------------------------------------
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
	int facesMatched[no_of_faces[index]]; 
	
	// NEED A LIST OF IOU TO PASS INTO TPR
	float IOUs[no_of_faces[index]];

	cout << "-------------------"<<endl<< "INDEX: " << index << std::endl;
	cout << "NO OF FACES actually: " << no_of_faces[index] << endl;
	// for each ground truth face we perform IOU with all detected faces
	for (int j = 0; j < no_of_faces[index]; j++){
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

	// TPR
	float tpr = calcTPR(IOUs, index);
	std::cout << "----- TPR: " << tpr << std::endl;

	// F1
	// no of detected should increment whenever one we dont include
	int noOfDetected = faces.size();
	cout << "detected: " << noOfDetected << endl;
	calcF1score(IOUs, index, noOfDetected);

}



/*
this diagram is useful for iou calculation below
l1----------
 |			|
 |	 l2-----|-------
  ---|-----r1		|
	 |				|
	 |				|
	  -------------r2

a legend of sorts:
px, py, pw, ph -> predicted data
gx, gy, gw, gh -> ground truth data
*/
float calcIOU(string fname, int px, int py, int pw, int ph, int col){
	cout << "COLE: "<< col << endl;
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
	for (int i = 0 ; i < no_of_faces[index]; i++){
		if (iou[i] >= 0.5 ){
			// then it is a true positivei
			TP = TP + 1;
		}
	}
	float tpr = (float) TP / no_of_faces[index];
	return tpr;
}


float calcF1score(float iou[], int index, int noOfDetected){
	int TP = 0;
	int FN = 0;
	int FP = noOfDetected - no_of_faces[index];

	if (FP < 0){
		FP = 0;
	}
	cout << endl;
	cout << "index: " << index << endl;
	for (int i = 0 ; i < no_of_faces[index]; i++){
	cout << "IOU VALUE: "<< iou[i] << endl;
		if (iou[i] >= 0.5 ){
			// then it is a true positive
			TP = TP + 1;
		}
		if (iou[i] == 0) {
			FN = FN + 1;
		}	
	}
	cout << "fp: " << FP << endl;
	cout << "tp: " << TP << endl;
	cout << "fn: " << FN << endl;
	// ACTUAL F1	
	float precision = (float) TP / (TP + FP);
	float recall = (float) TP / (TP + FN);
	cout<< "precision: " << precision << endl;
	cout << "recall: " << recall << endl;

	float f1; 
	f1 = 2 * ((precision * recall)/(precision + recall));

	cout<< "------ F1 SCORE: " <<  f1 << endl;;
	return f1;
}
