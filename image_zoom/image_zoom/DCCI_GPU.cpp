#include <algorithm>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "dcci_gpu.h"

using namespace std;
using namespace cv;

size_t numRows, numCols;
uchar3* d_in;

void loadImage(string& filename, uchar3** imagePtr, size_t* numRows, size_t* numCols)
{
	Mat image = imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);

	if (image.empty())
	{
		cerr << "Failed to load image" << filename << endl;
		exit(1);
	}

	if (image.channels() != 3)
	{
		cerr << "Image must be color!" << endl;
		exit(1);
	}

	if (!image.isContinuous())
	{
		cerr << "Image isn't continuous!" << endl;
		exit(1);
	}

	Mat Result(2 * image.rows, 2 * image.cols, CV_8UC3, Scalar(0, 0, 0));

	//set the i and j to the offset value
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			int ii = 2 * i;
			int jj = 2 * j;
			Vec3b pixel = image.at<Vec3b>(i, j);
			Result.at<Vec3b>(ii, jj) = pixel;
		}
	}
	
	*imagePtr = new uchar3[Result.rows*Result.cols];
	unsigned char* cvPtr = Result.ptr<unsigned char>(0);
	for (size_t i = 0; i < Result.rows*Result.cols; i++)
	{
		(*imagePtr)[i].x = cvPtr[3 * i + 0];
		(*imagePtr)[i].y = cvPtr[3 * i + 1];
		(*imagePtr)[i].z = cvPtr[3 * i + 2];
	}

	//use the original ones
	*numRows = Result.rows;
	*numCols = Result.cols;	
}

void saveImage(uchar3* image, string& filename, size_t numRows, size_t numCols)
{
	int sizes[2] = { numRows, numCols };
	Mat imageRGBA(2, sizes, CV_8UC3, (void*)image);
	imwrite(filename.c_str(), imageRGBA);
}

void load_image_in_GPU(string filename)
{
	uchar3* h_image;
	loadImage(filename, &h_image, &numRows, &numCols);
	cudaMalloc((void**)&d_in, numRows*numCols * sizeof(uchar3));
	cudaMemcpy(d_in, h_image, numRows*numCols * sizeof(uchar3), cudaMemcpyHostToDevice);
	free(h_image);
}


void main()
{
	load_image_in_GPU("D:/engle/image_zoom/image_zoom/lena.jpg");
	uchar3* h_out = NULL;

	h_out = dcci(d_in,numRows, numCols);
	cudaFree(d_in);
	string outputfile = "D:/engle/image_zoom/image_zoom/lena3.jpg";
	if (h_out != NULL)
		saveImage(h_out, outputfile, numRows, numCols);
}
