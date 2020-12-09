#include <stdio.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>
#include<iostream>
#include<vector>

using namespace std;
using namespace cv;

int main()
{
    Mat Gray_Image;
    Mat Blurred_Image;
    Mat Thresholded_Image;
    Mat matClassificationInts;
    Mat matTrainingImagesAsFlattenedFloats;
    vector<vector<Point>> Contours;
    vector<Vec4i> Hierarchy;

    vector<int> ValidChars;
    for (int i = 48; i <= 59; i++)
        ValidChars.push_back(i);
    for (int i = 65; i <= 90; i++)
        ValidChars.push_back(i);

    Mat Input_Image = imread("Test.png");

    cvtColor(Input_Image, Gray_Image, COLOR_BGR2GRAY);

    GaussianBlur(Gray_Image, Blurred_Image, Size(5, 5), 0);

    adaptiveThreshold(Blurred_Image, Thresholded_Image, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);

    Mat Image_Copy = Thresholded_Image.clone();

    findContours(Image_Copy, Contours, Hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < Contours.size(); i++)
    {
        if (contourArea(Contours[i]) > 100)
        {
            Rect BoundingRect = boundingRect(Contours[i]);

            rectangle(Input_Image, BoundingRect, Scalar(0, 0, 255), 2);

            Mat matROI = Thresholded_Image(BoundingRect);

            Mat matROIResized;
            Mat Input_Resized;
            resize(matROI, matROIResized, Size(20, 30));
            resize(Input_Image, Input_Resized, Size(1350, 700));


            imshow("Input Image", Input_Resized);
            imshow("Character", matROIResized);
            int intChar = waitKey(0);

            if (intChar == 27)
                break;



            Mat matImageFloat;
            matROIResized.convertTo(matImageFloat, CV_32FC1);

            Mat matImageFlattenedFloat = matImageFloat.reshape(1, 1);
            matClassificationInts.push_back(intChar);
            matTrainingImagesAsFlattenedFloats.push_back(matImageFlattenedFloat);
        }
    }

    FileStorage fsClassifications("classifications.xml", FileStorage::WRITE);
    fsClassifications << "classifications" << matClassificationInts;
    fsClassifications.release();

    FileStorage fsTrainingImages("images.xml", FileStorage::WRITE);
    fsTrainingImages << "images" << matTrainingImagesAsFlattenedFloats;
    fsTrainingImages.release();

    waitKey(0);
    return(0);
}
