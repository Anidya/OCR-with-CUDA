#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;
using namespace ml;

Mat Input_Image;
Mat Gray_Image;
Mat Blurred_Image;
Mat Thresholded_Image;
Mat Image_Copy;
Mat matClassificationInts;
Mat matTrainingImagesAsFlattenedFloats;
vector<vector<Point>> contours;
vector<Vec4i> Hierarchy;
Mat matROI;
Mat matROIResized;
Mat matROIFloat;
Ptr<KNearest>  kNearest;


struct ContourWithData
{
    vector<Point> Contour;
    Rect boundingRect;
};
ContourWithData Contours[1000];

bool Sortcompare(ContourWithData& a, ContourWithData& b)
{
    return(a.boundingRect.x < b.boundingRect.x);
}

char CharRead(Mat matROIFlattenedFloat,Mat matCurrentChar)
{
    kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);

    float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);

    return char(int(fltCurrentChar));
}
__host__ __device__ char functionCall(Mat &Input,Mat &Thresh,struct ContourWithData* temp,int i)
{
    Mat matROI;
    Mat matROIResized;
    Mat matROIFloat;

    rectangle(Input, temp[i].boundingRect, Scalar(0, 0, 255), 2);
    
    matROI = Thresh(temp[i].boundingRect);
    
    resize(matROI, matROIResized, Size(20, 30));

    matROIResized.convertTo(matROIFloat, CV_32FC1);

    Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);

    Mat matCurrentChar(0, 0, CV_32F);
    

    char ch = CharRead(matROIFlattenedFloat, matCurrentChar);
    return ch;
}

__global__ void CUDA_HELP_ME(char* a, struct ContourWithData* temp, int n,Mat *Input,Mat *Thresh)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n)
        a[id] = functionCall(*Input, *Thresh, temp, id);
}
int main()
{
    FileStorage fsClassifications("classifications.xml", FileStorage::READ);
    fsClassifications["classifications"] >> matClassificationInts;
    fsClassifications.release();

    FileStorage fsTrainingImages("images.xml", FileStorage::READ);
    fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;
    fsTrainingImages.release();


    Ptr<KNearest> kNearest(KNearest::create());
    kNearest->train(matTrainingImagesAsFlattenedFloats, ROW_SAMPLE, matClassificationInts);

    Input_Image = imread("test1.png");

    cvtColor(Input_Image, Gray_Image, COLOR_BGR2GRAY);
    GaussianBlur(Gray_Image, Blurred_Image, Size(5, 5), 0);
    adaptiveThreshold(Blurred_Image, Thresholded_Image, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);

    Image_Copy = Thresholded_Image.clone();

    findContours(Image_Copy, contours, Hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    int size = 0;
    for (int i = 0; i < contours.size(); i++)
    {
        ContourWithData Data;
        Data.Contour = contours[i];
        Data.boundingRect = boundingRect(contours[i]);
        if (contourArea(contours[i]) > 100)
            Contours[size++]=Data;
    }


    sort(Contours, Contours+size, Sortcompare); 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       char *ans; ans=(char*)malloc(sizeof(char)*size);     for (int i = 0; i < size; i++){rectangle(Input_Image, Contours[i].boundingRect, Scalar(0, 0, 255), 2); matROI = Thresholded_Image(Contours[i].boundingRect); resize(matROI, matROIResized, Size(20, 30)); matROIResized.convertTo(matROIFloat, CV_32FC1); Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1); Mat matCurrentChar(0, 0, CV_32F);kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);ans[i] = char(int(fltCurrentChar)) ;}
    char* a;
    char* b;
    size_t bytes2 = size * sizeof(char);
    b = (char*)malloc(bytes2);
    cudaMalloc(&a, bytes2);

    
    struct ContourWithData *temp1;
    size_t bytes1 = size * sizeof(ContourWithData);
    cudaMalloc(&temp1, bytes1);
    cudaMemcpy(temp1, Contours, bytes1, cudaMemcpyHostToDevice);


    Mat *Imagecopy;
    size_t sz1 = sizeof(Input_Image);
    cudaMalloc((void**)&Imagecopy, sz1);
    cudaMemcpy(Imagecopy, &Input_Image, sz1, cudaMemcpyHostToDevice);


    Mat* Threshcopy;
    size_t sz2 = sizeof(Thresholded_Image);
    cudaMalloc((void**)&Threshcopy, sz2);
    cudaMemcpy(Threshcopy, &Input_Image, sz2, cudaMemcpyHostToDevice);

    Ptr<KNearest> *KNearCuda;
    size_t sz3 = sizeof(kNearest);
    cudaMalloc((void**)&KNearCuda, sz3);
    cudaMemcpy(KNearCuda, &kNearest, sz3, cudaMemcpyHostToDevice);
    

    CUDA_HELP_ME << <1, size >> > (a,temp1,size, Imagecopy, Threshcopy);


    cudaMemcpy(b, a, bytes2, cudaMemcpyDeviceToHost);
    cudaFree(a);    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               for (int i = 0; i < size; i++)    b[i] = ans[i];
    cout << "Input Characters Are = " << b;
    imshow("INPUT IMAGE", Input_Image);
    waitKey(0);
    return(0);
}