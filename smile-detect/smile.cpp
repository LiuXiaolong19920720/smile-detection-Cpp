#include<opencv2\opencv.hpp>  
#include <iostream>  
#include <stdio.h>  

using namespace std;
using namespace cv;

String face_cascade_name = "haarcascade_frontalface_default.xml";
String smile_cascade_name = "haarcascade_smile.xml";
CascadeClassifier face_cascade;     
CascadeClassifier smile_cascade;   
String window_name = "Capture - Face detection";

int main()
{
	VideoCapture capture;  
	Mat frame;  

	if (!face_cascade.load(face_cascade_name))
	{
		printf("--(!)Error loading face cascade\n"); 
		return -1;
	};
	if (!smile_cascade.load(smile_cascade_name)) 
	{
		printf("--(!)Error loading eyes cascade\n"); 
		return -1;
	};

	//-- 2. Read the video stream  
	capture.open(0);  
	if (!capture.isOpened()) 
	{ 
		printf("--(!)Error opening video capture\n"); 
		return -1; 
	}  

	while (capture.read(frame))
	{
		if (frame.empty())
		{
			printf(" --(!) No captured frame -- Break!");
			break;
		}

		std::vector<Rect> faces;
		Mat frame_gray;

		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);
		face_cascade.detectMultiScale(frame_gray, faces, 1.05, 8, CASCADE_SCALE_IMAGE);

		for (size_t i = 0; i < faces.size(); i++)
		{
			rectangle(frame, faces[i], Scalar(255, 0, 0), 2, 8, 0);

			Mat faceROI = frame_gray(faces[i]);
			std::vector<Rect> smile;

			//-- In each face, detect smile
			smile_cascade.detectMultiScale(faceROI, smile, 1.1, 55, CASCADE_SCALE_IMAGE);

			for (size_t j = 0; j < smile.size(); j++)
			{
				Rect rect(faces[i].x + smile[j].x, faces[i].y + smile[j].y, smile[j].width, smile[j].height);
				rectangle(frame, rect, Scalar(0, 0, 255), 2, 8, 0);
			}
		}
		//-- Show what you got  
		namedWindow(window_name, 2);
		imshow(window_name, frame);
		waitKey(100);
	}
	int c = waitKey(0);
	if ((char)c == 27) { return 0; } 

	return 0;
}