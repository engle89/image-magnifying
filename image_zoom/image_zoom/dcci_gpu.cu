#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "cutil_math.h"

uchar3* d_inputImageRGBA__;
uchar3* d_outputImageRGBA1__;
uchar3* d_outputImageRGBA2__;
uchar3* d_outputImageRGBA__;

__device__ float attract(float x, float y, float z, float t, float expose)
{
	expose = 1. / 2.;
	return pow((-2 * pow(x, expose) + 10 * pow(y, expose) + 10 * pow(z, expose) - 2 * pow(t, expose)) / 16., 1. / expose);
}

__global__ void firstpass(const uchar3* const inputImage,
	                      uchar3* const outputImage,
	                      int numRows,
	                      int numCols)
{
	float threshold = 1.15;
	int k = 5;

	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	if (col >= numCols || row >= numRows)
		return;

	int out = row * numCols + col;

	if (col % 2 == 0)
	{
		if (row % 2 == 0)
		{
			outputImage[out] = inputImage[out];
			return;
		}
		else
			return;
	}
	else
	{
		if (row % 2 == 0)
			return;
	}
	
	float G1_b = 0;
	float G2_b = 0;
	float G1_g = 0;
	float G2_g = 0;
	float G1_r = 0;
	float G2_r = 0;
	int temp1;
	int temp2;

	
	int Row = (row - 1) / 2;
	int Col = (col - 1) / 2;
	for (int x = -1; x < 2; x++)
	{
		for (int y = -1; y < 2; y++)
		{
			//G1:
			float pix11;
			float pix12;
			float pix21;
			float pix22;

			temp1 = 2 * (Row + x + 1) * numCols + 2 * (Col + y);
			temp2 = 2 * (Row + x) * numCols + 2 * (Col + y + 1);

			if (Row == 0)
			{
				if (Col == 0)
				{
					pix11 = 0;
					pix12 = 0;
				}
				else
				{
					pix11 = static_cast<float>(inputImage[temp1].x);
					pix12 = 0;
				}
			}
			else
			{
				if (Col == 0)
				{
					pix11 = 0;
					pix12 = static_cast<float>(inputImage[temp2].x);
				}
				else
				{
					pix11 = static_cast<float>(inputImage[temp1].x);
					pix12 = static_cast<float>(inputImage[temp2].x);
				}
			}
			G1_b += abs(pix11 - pix12);

			if (Row == 0)
			{
				if (Col == 0)
				{
					pix11 = 0;
					pix12 = 0;
				}
				else
				{
					pix11 = static_cast<float>(inputImage[temp1].y);
					pix12 = 0;
				}
			}
			else
			{
				if (Col == 0)
				{
					pix11 = 0;
					pix12 = static_cast<float>(inputImage[temp2].y);
				}
				else
				{
					pix11 = static_cast<float>(inputImage[temp1].y);
					pix12 = static_cast<float>(inputImage[temp2].y);
				}
			}
			G1_g += abs(pix11 - pix12);

			if (Row == 0)
			{
				if (Col == 0)
				{
					pix11 = 0;
					pix12 = 0;
				}
				else
				{
					pix11 = static_cast<float>(inputImage[temp1].z);
					pix12 = 0;
				}
			}
			else
			{
				if (Col == 0)
				{
					pix11 = 0;
					pix12 = static_cast<float>(inputImage[temp2].z);
				}
				else
				{
					pix11 = static_cast<float>(inputImage[temp1].z);
					pix12 = static_cast<float>(inputImage[temp2].z);
				}
			}
			G1_r += abs(pix11 - pix12);

			//G2:
			temp1 = 2 * (Row + x) * numCols + 2 * (Col + y);
			temp2 = 2 * (Row + x + 1) * 2 * numCols + 2 * (Col + y + 1);

			if (Row == 0)
			{
				pix11 = 0;
				pix12 = static_cast<float>(inputImage[temp2].x);
			}
			else
			{
				if (Col == 0)
				{
					pix21 = 0;
					pix22 = static_cast<float>(inputImage[temp2].x);
				}
				else
				{
					pix21 = static_cast<float>(inputImage[temp1].x);
					pix22 = static_cast<float>(inputImage[temp2].x);
				}
			}
			G2_b += abs(pix21 - pix22);

			if (Row == 0)
			{
				pix11 = 0;
				pix12 = static_cast<float>(inputImage[temp2].y);
			}
			else
			{
				if (Col == 0)
				{
					pix21 = 0;
					pix22 = static_cast<float>(inputImage[temp2].y);
				}
				else
				{
					pix21 = static_cast<float>(inputImage[temp1].y);
					pix22 = static_cast<float>(inputImage[temp2].y);
				}
			}
			G2_g += abs(pix21 - pix22);

			if (Row == 0)
			{
				pix11 = 0;
				pix12 = static_cast<float>(inputImage[temp2].z);
			}
			else
			{
				if (Col == 0)
				{
					pix21 = 0;
					pix22 = static_cast<float>(inputImage[temp2].z);
				}
				else
				{
					pix21 = static_cast<float>(inputImage[temp1].z);
					pix22 = static_cast<float>(inputImage[temp2].z);
				}
			}
			G2_r += abs(pix21 - pix22);
		}
	}

	int ipix00 = 2 * (Row - 1) * numCols + 2 * (Col - 1);
	int ipix11 = 2 * Row * numCols + 2 * Col;
	int ipix22 = 2 * (Row + 1) * numCols + 2 * (Col + 1);
	int ipix33 = 2 * (Row + 2) * numCols + 2 * (Col + 2);
	int ipix30 = 2 * (Row + 2) * numCols + 2 * (Col - 1);
	int ipix21 = 2 * (Row + 1) * numCols + 2 * Col;
	int ipix12 = 2 * Row * numCols + 2 * (Col + 1);
	int ipix03 = 2 * (Row - 1) * numCols + 2 * (Col + 2);

	uchar3 pix00;
	uchar3 pix11;
	uchar3 pix22;
	uchar3 pix33;
	uchar3 pix30;
	uchar3 pix21;
	uchar3 pix12;
	uchar3 pix03;

	if (Row == 0)
	{
		if (Col == 0)
		{
			pix00 = make_uchar3(0, 0, 0);
			pix30 = make_uchar3(0, 0, 0);
			pix03 = make_uchar3(0, 0, 0);
		}
		else
		{
			pix00 = make_uchar3(0, 0, 0);
			pix30 = inputImage[ipix30];
			pix03 = make_uchar3(0, 0, 0);
		}
	}
	else
	{
		if (Col == 0)
		{
			pix00 = make_uchar3(0, 0, 0);
			pix30 = make_uchar3(0, 0, 0);
			pix03 = inputImage[ipix03];
		}
		else
		{
			pix00 = inputImage[ipix00];
			pix30 = inputImage[ipix30];
			pix03 = inputImage[ipix03];
		}
	}
	pix11 = inputImage[ipix11];
	pix22 = inputImage[ipix22];
	pix33 = inputImage[ipix33];
	pix21 = inputImage[ipix21];
	pix12 = inputImage[ipix12];

	float output_b;
	float output_g;
	float output_r;
	//b
	if (100 * (1 + G1_b) > 100 * threshold*(1 + G2_b))
	{
		float val = attract(static_cast<float>(pix00.x), static_cast<float>(pix11.x), static_cast<float>(pix22.x), static_cast<float>(pix33.x), 1.);
		if (val < 0)
			val = 0;
		if (val > 255)
		    val = 255;
		output_b = val;
	}
	else if (100 * (1 + G2_b) > 100 * threshold*(1 + G1_b))
	{
		float val = attract(static_cast<float>(pix30.x), static_cast<float>(pix21.x), static_cast<float>(pix12.x), static_cast<float>(pix03.x), 1.);
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_b = val;
	}
	else
	{
		float w1 = 1 / (1 + pow(G1_b, k));
		float w2 = 1 / (1 + pow(G2_b, k));
		float weight1 = w1 / (w1 + w2);
		float weight2 = w2 / (w1 + w2);
		float p1 = attract(static_cast<float>(pix00.x), static_cast<float>(pix11.x), static_cast<float>(pix22.x), static_cast<float>(pix33.x), 1. / 6.);
		float p2 = attract(static_cast<float>(pix30.x), static_cast<float>(pix21.x), static_cast<float>(pix12.x), static_cast<float>(pix03.x), 1. / 6.);
		float val = weight1 * p1 + weight2 * p2;
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_b = val;
	}

	//g
	if (100 * (1 + G1_g) > 100 * threshold*(1 + G2_g))
	{
		float val = attract(static_cast<float>(pix00.y), static_cast<float>(pix11.y), static_cast<float>(pix22.y), static_cast<float>(pix33.y), 1.);
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_g = val;
	}
	else if (100 * (1 + G2_g) > 100 * threshold*(1 + G1_g))
	{
		float val = attract(static_cast<float>(pix30.y), static_cast<float>(pix21.y), static_cast<float>(pix12.y), static_cast<float>(pix03.y), 1.);
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_g = val;
	}
	else
	{
		float w1 = 1 / (1 + pow(G1_g, k));
		float w2 = 1 / (1 + pow(G2_g, k));
		float weight1 = w1 / (w1 + w2);
		float weight2 = w2 / (w1 + w2);
		float p1 = attract(static_cast<float>(pix00.y), static_cast<float>(pix11.y), static_cast<float>(pix22.y), static_cast<float>(pix33.y), 1. / 6.);
		float p2 = attract(static_cast<float>(pix30.y), static_cast<float>(pix21.y), static_cast<float>(pix12.y), static_cast<float>(pix03.y), 1. / 6.);
		float val = weight1 * p1 + weight2 * p2;
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_g = val;
	}

	//r
	if (100 * (1 + G1_r) > 100 * threshold*(1 + G2_r))
	{
		float val = attract(static_cast<float>(pix00.z), static_cast<float>(pix11.z), static_cast<float>(pix22.z), static_cast<float>(pix33.z), 1.);
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_r = val;
	}
	else if (100 * (1 + G2_r) > 100 * threshold*(1 + G1_r))
	{
		float val = attract(static_cast<float>(pix30.z), static_cast<float>(pix21.z), static_cast<float>(pix12.z), static_cast<float>(pix03.z), 1.);
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_r = val;
	}
	else
	{
		float w1 = 1 / (1 + pow(G1_r, k));
		float w2 = 1 / (1 + pow(G2_r, k));
		float weight1 = w1 / (w1 + w2);
		float weight2 = w2 / (w1 + w2);
		float p1 = attract(static_cast<float>(pix00.z), static_cast<float>(pix11.z), static_cast<float>(pix22.z), static_cast<float>(pix33.z), 1. / 6.);
		float p2 = attract(static_cast<float>(pix30.z), static_cast<float>(pix21.z), static_cast<float>(pix12.z), static_cast<float>(pix03.z), 1. / 6.);
		float val = weight1 * p1 + weight2 * p2;
		if (val < 0)
			val = 0;
		if (val > 255)
		    val = 255;
		output_r = val;
	}

	outputImage[out].x = output_b;
	outputImage[out].y = output_g;
	outputImage[out].z = output_r;
}

__global__ void secondpass1(const uchar3* const inputImage,
	                        uchar3* const outputImage,
	                        int numRows,
	                        int numCols)
{
	double threshold = 1.15;
	int k = 5;

	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	if (col >= numCols || row > numRows)
		return;

	int out = row * numCols + col;

	if (row % 2 == 0)
	{
		outputImage[out] = inputImage[out];
		return;
	}
	else
	{
		if (col % 2 == 1)
		{
			outputImage[out] = inputImage[out];
			return;
		}
	}

	int x = row;
	int y = col;

	int isquare_left1   = (x - 1) * numCols + (y - 2);
	int isquare_left2   = (x + 1) * numCols + (y - 2);
	int icircle_left1   = (x - 2) * numCols + (y - 1);
	int icircle_left2   = (x    ) * numCols + (y - 1);
	int icircle_left3   = (x + 2) * numCols + (y - 1);
	int isquare_middle1 = (x - 1) * numCols + (y    );
	int isquare_middle2 = (x + 1) * numCols + (y    );
	int icircle_right1  = (x - 2) * numCols + (y + 1);
	int icircle_right2  = (x    ) * numCols + (y + 1);
	int icircle_right3  = (x + 2) * numCols + (y + 1);
	int isquare_right1  = (x - 1) * numCols + (y + 2);
	int isquare_right2  = (x + 1) * numCols + (y + 2);
	int imore1          = (x - 3) * numCols + (y    );
	int imore2          = (x + 3) * numCols + (y    );
	int imore3          = (x    ) * numCols + (y + 3);
	int imore4          = (x    ) * numCols + (y - 3);

	uchar3 square_left1;
	uchar3 square_left2;
	uchar3 circle_left1;
	uchar3 circle_left2;
	uchar3 circle_left3;
	uchar3 square_middle1;
	uchar3 square_middle2;
	uchar3 circle_right1;
	uchar3 circle_right2;
	uchar3 circle_right3;
	uchar3 square_right1;
	uchar3 square_right2;
    uchar3 more1;
	uchar3 more2;
	uchar3 more3;
	uchar3 more4;

	if (x == 0)
	{
		if (y == 0)
		{
			square_left1   = make_uchar3(0, 0, 0);
			square_left2   = make_uchar3(0, 0, 0);
			circle_left1   = make_uchar3(0, 0, 0);
			circle_left2   = make_uchar3(0, 0, 0);
			circle_left3   = make_uchar3(0, 0, 0);
			square_middle1 = make_uchar3(0, 0, 0);
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = make_uchar3(0, 0, 0);
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = make_uchar3(0, 0, 0);
			square_right2  = inputImage[isquare_right2];
			more1          = make_uchar3(0, 0, 0);
			more2          = inputImage[imore2];
			more3          = inputImage[imore3];
			more4          = make_uchar3(0, 0, 0);
		}
		else if (y == 1)
		{
			square_left1   = make_uchar3(0, 0, 0);
			square_left2   = make_uchar3(0, 0, 0);
			circle_left1   = make_uchar3(0, 0, 0);
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = make_uchar3(0, 0, 0);
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = make_uchar3(0, 0, 0);
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = make_uchar3(0, 0, 0);
			square_right2 = inputImage[isquare_right2];
			more1         = make_uchar3(0, 0, 0);
			more2         = inputImage[imore2];
			more3         = inputImage[imore3];
			more4         = make_uchar3(0, 0, 0);
		}
		else if (y == 2)
		{
			square_left1 = make_uchar3(0, 0, 0);
			square_left2 = inputImage[isquare_left2];
			circle_left1 = make_uchar3(0, 0, 0);
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = make_uchar3(0, 0, 0);
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = make_uchar3(0, 0, 0);
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = make_uchar3(0, 0, 0);
			square_right2 = inputImage[isquare_right2];
			more1         = make_uchar3(0, 0, 0);
			more2         = inputImage[imore2];
			more3         = inputImage[imore3];
			more4         = make_uchar3(0, 0, 0);
		}
		else if (y == numCols - 3)
		{
			square_left1   = make_uchar3(0, 0, 0);
			square_left2   = inputImage[isquare_left2];
			circle_left1   = make_uchar3(0, 0, 0);
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = make_uchar3(0, 0, 0);
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = make_uchar3(0, 0, 0);
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = make_uchar3(0, 0, 0);
			square_right2 = inputImage[isquare_right2];
			more1         = make_uchar3(0, 0, 0);
			more2         = inputImage[imore2];
			more3         = make_uchar3(0, 0, 0);
			more4         = inputImage[imore4];
		}
		else if (y == numCols - 2)
		{
			square_left1   = make_uchar3(0, 0, 0);
			square_left2   = inputImage[isquare_left2];
			circle_left1   = make_uchar3(0, 0, 0);
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = make_uchar3(0, 0, 0);
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = make_uchar3(0, 0, 0);
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = make_uchar3(0, 0, 0);
			square_right2  = make_uchar3(0, 0, 0);
			more1          = make_uchar3(0, 0, 0);
			more2          = inputImage[imore2];
			more3          = make_uchar3(0, 0, 0);
			more4          = inputImage[imore4];
		}
		else if (y == numCols - 1)
		{
			square_left1   = make_uchar3(0, 0, 0);
			square_left2   = inputImage[isquare_left2];
			circle_left1   = make_uchar3(0, 0, 0);
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = make_uchar3(0, 0, 0);
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = make_uchar3(0, 0, 0);
			circle_right2  = make_uchar3(0, 0, 0);
			circle_right3  = make_uchar3(0, 0, 0);
			square_right1  = make_uchar3(0, 0, 0);
			square_right2  = make_uchar3(0, 0, 0);
			more1          = make_uchar3(0, 0, 0);
			more2          = inputImage[imore2];
			more3          = make_uchar3(0, 0, 0);
			more4          = inputImage[imore4];
		}
		else
		{
			square_left1   = make_uchar3(0, 0, 0);
			square_left2   = inputImage[isquare_left2];
			circle_left1   = make_uchar3(0, 0, 0);
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = make_uchar3(0, 0, 0);
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = make_uchar3(0, 0, 0);
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = make_uchar3(0, 0, 0);
			square_right2  = inputImage[isquare_right2];
			more1          = make_uchar3(0, 0, 0);
			more2          = inputImage[imore2];
			more3          = inputImage[imore3];
			more4          = inputImage[imore4];
		}
	}
	else if (x == 1)
	{
		if (y == 0)
		{
			square_left1   = make_uchar3(0, 0, 0);
			square_left2   = make_uchar3(0, 0, 0);
			circle_left1   = make_uchar3(0, 0, 0);
			circle_left2   = make_uchar3(0, 0, 0);
			circle_left3   = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = make_uchar3(0, 0, 0);
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1         = make_uchar3(0, 0, 0);
			more2         = inputImage[imore2];
			more3         = inputImage[imore3];
			more4         = make_uchar3(0, 0, 0);
		}
		else if (y == 1)
		{  
			square_left1   = make_uchar3(0, 0, 0);
			square_left2   = make_uchar3(0, 0, 0);
			circle_left1   = make_uchar3(0, 0, 0);
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = make_uchar3(0, 0, 0);
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1         = make_uchar3(0, 0, 0);
			more2         = inputImage[imore2];
			more3         = inputImage[imore3];
			more4         = make_uchar3(0, 0, 0);
		}
		else if (y == 2)
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = make_uchar3(0, 0, 0);
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = make_uchar3(0, 0, 0);
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = inputImage[isquare_right1];
			square_right2  = inputImage[isquare_right2];
			more1          = make_uchar3(0, 0, 0);
			more2          = inputImage[imore2];
			more3          = inputImage[imore3];
			more4          = make_uchar3(0, 0, 0);
		}
		else if (y == numCols - 3)
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = make_uchar3(0, 0, 0);
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = make_uchar3(0, 0, 0);
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = inputImage[isquare_right1];
			square_right2  = inputImage[isquare_right2];
			more1          = make_uchar3(0, 0, 0);
			more2          = inputImage[imore2];
			more3          = make_uchar3(0, 0, 0);
			more4          = inputImage[imore4];
		}
		else if (y == numCols - 2)
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = make_uchar3(0, 0, 0);
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = make_uchar3(0, 0, 0);
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = make_uchar3(0, 0, 0);
			square_right2  = make_uchar3(0, 0, 0);
			more1          = make_uchar3(0, 0, 0);
			more2          = inputImage[imore2];
			more3          = make_uchar3(0, 0, 0);
			more4          = inputImage[imore4];
		}
		else if (y == numCols - 1)
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = make_uchar3(0, 0, 0);
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = make_uchar3(0, 0, 0);
			circle_right2  = make_uchar3(0, 0, 0);
			circle_right3  = make_uchar3(0, 0, 0);
			square_right1  = make_uchar3(0, 0, 0);
			square_right2  = make_uchar3(0, 0, 0);
			more1          = make_uchar3(0, 0, 0);
			more2          = inputImage[imore2];
			more3          = make_uchar3(0, 0, 0);
			more4          = inputImage[imore4];
		}
		else
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = make_uchar3(0, 0, 0);
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = make_uchar3(0, 0, 0);
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = inputImage[isquare_right1];
			square_right2  = inputImage[isquare_right2];
			more1          = make_uchar3(0, 0, 0);
			more2          = inputImage[imore2];
			more3          = inputImage[imore3];
			more4          = inputImage[imore4];
		}
	}
	else if (x == 2)
	{
		if (y == 0)
		{
			square_left1   = make_uchar3(0, 0, 0);
			square_left2   = make_uchar3(0, 0, 0);
			circle_left1   = make_uchar3(0, 0, 0);
			circle_left2   = make_uchar3(0, 0, 0);
			circle_left3   = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = inputImage[isquare_right1];
			square_right2  = inputImage[isquare_right2];
			more1          = make_uchar3(0, 0, 0);
			more2          = inputImage[imore2];
			more3          = inputImage[imore3];
			more4          = make_uchar3(0, 0, 0);
		}
		else if (y == 1)
		{
			square_left1   = make_uchar3(0, 0, 0);
			square_left2   = make_uchar3(0, 0, 0);
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = inputImage[isquare_right1];
			square_right2  = inputImage[isquare_right2];
			more1          = make_uchar3(0, 0, 0);
			more2          = inputImage[imore2];
			more3          = inputImage[imore3];
			more4          = make_uchar3(0, 0, 0);
		}
		else if (y == 2)
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = inputImage[isquare_right1];
			square_right2  = inputImage[isquare_right2];
			more1          = make_uchar3(0, 0, 0);
			more2          = inputImage[imore2];
			more3          = inputImage[imore3];
			more4          = make_uchar3(0, 0, 0);
		}
		else if (y == numCols - 3)
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = inputImage[isquare_right1];
			square_right2  = inputImage[isquare_right2];
			more1          = make_uchar3(0, 0, 0);
			more2          = inputImage[imore2];
			more3          = make_uchar3(0, 0, 0);
			more4          = inputImage[imore4];
		}
		else if (y == numCols - 2)
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = make_uchar3(0, 0, 0);
			square_right2  = make_uchar3(0, 0, 0);
			more1          = make_uchar3(0, 0, 0);
			more2          = inputImage[imore2];
			more3          = make_uchar3(0, 0, 0);
			more4          = inputImage[imore4];
		}
		else if (y == numCols - 1)
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = make_uchar3(0, 0, 0);
			circle_right2  = make_uchar3(0, 0, 0);
			circle_right3  = make_uchar3(0, 0, 0);
			square_right1  = make_uchar3(0, 0, 0);
			square_right2  = make_uchar3(0, 0, 0);
			more1          = make_uchar3(0, 0, 0);
			more2          = inputImage[imore2];
			more3          = make_uchar3(0, 0, 0);
			more4          = inputImage[imore4];
		}
		else
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = inputImage[isquare_right1];
			square_right2  = inputImage[isquare_right2];
			more1          = make_uchar3(0, 0, 0);
			more2          = inputImage[imore2];
			more3          = inputImage[imore3];
			more4          = inputImage[imore4];
		}
	}
	else if (x == numRows - 3)
	{
		if (y == 0)
		{
			square_left1   = make_uchar3(0, 0, 0);
			square_left2   = make_uchar3(0, 0, 0);
			circle_left1   = make_uchar3(0, 0, 0);
			circle_left2   = make_uchar3(0, 0, 0);
			circle_left3   = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = inputImage[isquare_right1];
			square_right2  = inputImage[isquare_right2];
			more1          = inputImage[imore1];
			more2          = make_uchar3(0, 0, 0);
			more3          = inputImage[imore3];
			more4          = make_uchar3(0, 0, 0);
		}
		else if (y == 1)
		{
			square_left1   = make_uchar3(0, 0, 0);
			square_left2   = make_uchar3(0, 0, 0);
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = inputImage[isquare_right1];
			square_right2  = inputImage[isquare_right2];
			more1          = inputImage[imore1];
			more2          = make_uchar3(0, 0, 0);
			more3          = inputImage[imore3];
			more4          = make_uchar3(0, 0, 0);
		}
		else if (y == 2)
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = inputImage[isquare_right1];
			square_right2  = inputImage[isquare_right2];
			more1          = inputImage[imore1];
			more2          = make_uchar3(0, 0, 0);
			more3          = inputImage[imore3];
			more4          = make_uchar3(0, 0, 0);
		}
		else if (y == numCols - 3)
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = inputImage[isquare_right1];
			square_right2  = inputImage[isquare_right2];
			more1          = inputImage[imore1];
			more2          = make_uchar3(0, 0, 0);
			more3          = make_uchar3(0, 0, 0);
			more4          = inputImage[imore4];
		}
		else if (y == numCols - 2)
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = make_uchar3(0, 0, 0);
			square_right2  = make_uchar3(0, 0, 0);
			more1          = inputImage[imore1];
			more2          = make_uchar3(0, 0, 0);
			more3          = make_uchar3(0, 0, 0);
			more4          = inputImage[imore4];
		}
		else if (y == numCols - 1)
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = make_uchar3(0, 0, 0);
			circle_right2  = make_uchar3(0, 0, 0);
			circle_right3  = make_uchar3(0, 0, 0);
			square_right1  = make_uchar3(0, 0, 0);
			square_right2  = make_uchar3(0, 0, 0);
			more1          = inputImage[imore1];
			more2          = make_uchar3(0, 0, 0);
			more3          = make_uchar3(0, 0, 0);
			more4          = inputImage[imore4];
		}
		else
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = inputImage[isquare_right1];
			square_right2  = inputImage[isquare_right2];
			more1          = inputImage[imore1];
			more2          = make_uchar3(0, 0, 0);
			more3          = inputImage[imore3];
			more4          = inputImage[imore4];
		}
	}
	else if (x == numRows - 2)
	{
		if (y == 0)
		{
			square_left1   = make_uchar3(0, 0, 0);
			square_left2   = make_uchar3(0, 0, 0);
			circle_left1   = make_uchar3(0, 0, 0);
			circle_left2   = make_uchar3(0, 0, 0);
			circle_left3   = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = make_uchar3(0, 0, 0);
			square_right1  = inputImage[isquare_right1];
			square_right2  = inputImage[isquare_right2];
			more1          = inputImage[imore1];
			more2          = make_uchar3(0, 0, 0);
			more3          = inputImage[imore3];
			more4          = make_uchar3(0, 0, 0);
		}
		else if (y == 1)
		{
			square_left1   = make_uchar3(0, 0, 0);
			square_left2   = make_uchar3(0, 0, 0);
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = make_uchar3(0, 0, 0);
			square_right1  = inputImage[isquare_right1];
			square_right2  = inputImage[isquare_right2];
			more1          = inputImage[imore1];
			more2          = make_uchar3(0, 0, 0);
			more3          = inputImage[imore3];
			more4          = make_uchar3(0, 0, 0);
		}
		else if (y == 2)
		{ 
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = make_uchar3(0, 0, 0);
			square_right1  = inputImage[isquare_right1];
			square_right2  = inputImage[isquare_right2];
			more1          = inputImage[imore1];
			more2          = make_uchar3(0, 0, 0);
			more3          = inputImage[imore3];
			more4          = make_uchar3(0, 0, 0);
		}
		else if (y == numCols - 3)
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = make_uchar3(0, 0, 0);
			square_right1  = inputImage[isquare_right1];
			square_right2  = inputImage[isquare_right2];
			more1          = inputImage[imore1];
			more2          = make_uchar3(0, 0, 0);
			more3          = make_uchar3(0, 0, 0);
			more4          = inputImage[imore4];
		}
		else if (y == numCols - 2)
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = make_uchar3(0, 0, 0);
			square_right1  = make_uchar3(0, 0, 0);
			square_right2  = make_uchar3(0, 0, 0);
			more1          = inputImage[imore1];
			more2          = make_uchar3(0, 0, 0);
			more3          = make_uchar3(0, 0, 0);
			more4          = inputImage[imore4];
		}
		else if (y == numCols - 1)
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = make_uchar3(0, 0, 0);
			circle_right2  = make_uchar3(0, 0, 0);
			circle_right3  = make_uchar3(0, 0, 0);
			square_right1  = make_uchar3(0, 0, 0);
			square_right2  = make_uchar3(0, 0, 0);
			more1          = inputImage[imore1];
			more2          = make_uchar3(0, 0, 0);
			more3          = make_uchar3(0, 0, 0);
			more4          = inputImage[imore4];
		}
		else
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = make_uchar3(0, 0, 0);
			square_right1  = inputImage[isquare_right1];
			square_right2  = inputImage[isquare_right2];
			more1          = inputImage[imore1];
			more2          = make_uchar3(0, 0, 0);
			more3          = inputImage[imore3];
			more4          = inputImage[imore4];
		}
	}
	else if (x == numRows - 1)
	{
		if (y == 0)
		{
			square_left1   = make_uchar3(0, 0, 0);
			square_left2   = make_uchar3(0, 0, 0);
			circle_left1   = make_uchar3(0, 0, 0);
			circle_left2   = make_uchar3(0, 0, 0);
			circle_left3   = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = make_uchar3(0, 0, 0);
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = make_uchar3(0, 0, 0);
			square_right1  = inputImage[isquare_right1];
			square_right2  = make_uchar3(0, 0, 0);
			more1          = inputImage[imore1];
			more2          = make_uchar3(0, 0, 0);
			more3          = inputImage[imore3];
			more4          = make_uchar3(0, 0, 0);
		}
		else if (y == 1)
		{
			square_left1   = make_uchar3(0, 0, 0);
			square_left2   = make_uchar3(0, 0, 0);
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = make_uchar3(0, 0, 0);
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = make_uchar3(0, 0, 0);
			square_right1  = inputImage[isquare_right1];
			square_right2  = make_uchar3(0, 0, 0);
			more1          = inputImage[imore1];
			more2          = make_uchar3(0, 0, 0);
			more3          = inputImage[imore3];
			more4          = make_uchar3(0, 0, 0);
		}
		else if (y == 2)
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = make_uchar3(0, 0, 0);
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = make_uchar3(0, 0, 0);
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = make_uchar3(0, 0, 0);
			square_right1  = inputImage[isquare_right1];
			square_right2  = make_uchar3(0, 0, 0);
			more1          = inputImage[imore1];
			more2          = make_uchar3(0, 0, 0);
			more3          = inputImage[imore3];
			more4          = make_uchar3(0, 0, 0);
		}
		else if (y == numCols - 3)
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = make_uchar3(0, 0, 0);
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = make_uchar3(0, 0, 0);
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = make_uchar3(0, 0, 0);
			square_right1  = inputImage[isquare_right1];
			square_right2  = make_uchar3(0, 0, 0);
			more1          = inputImage[imore1];
			more2          = make_uchar3(0, 0, 0);
			more3          = make_uchar3(0, 0, 0);
			more4          = inputImage[imore4];
		}
		else if (y == numCols - 2)
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = make_uchar3(0, 0, 0);
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = make_uchar3(0, 0, 0);
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = make_uchar3(0, 0, 0);
			square_right1  = make_uchar3(0, 0, 0);
			square_right2  = make_uchar3(0, 0, 0);
			more1          = inputImage[imore1];
			more2          = make_uchar3(0, 0, 0);
			more3          = make_uchar3(0, 0, 0);
			more4          = inputImage[imore4];
		}
		else if (y == numCols - 1)
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = make_uchar3(0, 0, 0);
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = make_uchar3(0, 0, 0);
			circle_right1  = make_uchar3(0, 0, 0);
			circle_right2  = make_uchar3(0, 0, 0);
			circle_right3  = make_uchar3(0, 0, 0);
			square_right1  = make_uchar3(0, 0, 0);
			square_right2  = make_uchar3(0, 0, 0);
			more1          = inputImage[imore1];
			more2          = make_uchar3(0, 0, 0);
			more3          = make_uchar3(0, 0, 0);
			more4          = inputImage[imore4];
		}
		else
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = make_uchar3(0, 0, 0);
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = make_uchar3(0, 0, 0);
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = make_uchar3(0, 0, 0);
			square_right1  = inputImage[isquare_right1];
			square_right2  = make_uchar3(0, 0, 0);
			more1          = inputImage[imore1];
			more2          = make_uchar3(0, 0, 0);
			more3          = inputImage[imore3];
			more4          = inputImage[imore4];
		}
	}
	else
	{
		if (y == 0)
		{
			square_left1   = make_uchar3(0, 0, 0);
			square_left2   = make_uchar3(0, 0, 0);
			circle_left1   = make_uchar3(0, 0, 0);
			circle_left2   = make_uchar3(0, 0, 0);
			circle_left3   = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = inputImage[isquare_right1];
			square_right2  = inputImage[isquare_right2];
			more1          = inputImage[imore1];
			more2          = inputImage[imore2];
			more3          = inputImage[imore3];
			more4          = make_uchar3(0, 0, 0);
		}
		else if (y == 1)
		{
			square_left1   = make_uchar3(0, 0, 0);
			square_left2   = make_uchar3(0, 0, 0);
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = inputImage[isquare_right1];
			square_right2  = inputImage[isquare_right2];
			more1          = inputImage[imore1];
			more2          = inputImage[imore2];
			more3          = inputImage[imore3];
			more4          = make_uchar3(0, 0, 0);
		}
		else if (y == 2)
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = inputImage[isquare_right1];
			square_right2  = inputImage[isquare_right2];
			more1          = inputImage[imore1];
			more2          = inputImage[imore2];
			more3          = inputImage[imore3];
			more4          = make_uchar3(0, 0, 0);
		}
		else if (y == numCols - 3)
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = inputImage[isquare_right1];
			square_right2  = inputImage[isquare_right2];
			more1          = inputImage[imore1];
			more2          = inputImage[imore2];
			more3          = make_uchar3(0, 0, 0);
			more4          = inputImage[imore4];
		}
		else if (y == numCols - 2)
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = make_uchar3(0, 0, 0);
			square_right2  = make_uchar3(0, 0, 0);
			more1          = inputImage[imore1];
			more2          = inputImage[imore2];
			more3          = make_uchar3(0, 0, 0);
			more4          = inputImage[imore4];
		}
		else if (y == numCols - 1)
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = make_uchar3(0, 0, 0);
			circle_right2  = make_uchar3(0, 0, 0);
			circle_right3  = make_uchar3(0, 0, 0);
			square_right1  = make_uchar3(0, 0, 0);
			square_right2  = make_uchar3(0, 0, 0);
			more1          = inputImage[imore1];
			more2          = inputImage[imore2];
			more3          = make_uchar3(0, 0, 0);
			more4          = inputImage[imore4];
		}
		else
		{
			square_left1   = inputImage[isquare_left1];
			square_left2   = inputImage[isquare_left2];
			circle_left1   = inputImage[icircle_left1];
			circle_left2   = inputImage[icircle_left2];
			circle_left3   = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1  = inputImage[icircle_right1];
			circle_right2  = inputImage[icircle_right2];
			circle_right3  = inputImage[icircle_right3];
			square_right1  = inputImage[isquare_right1];
			square_right2  = inputImage[isquare_right2];
			more1          = inputImage[imore1];
			more2          = inputImage[imore2];
			more3          = inputImage[imore3];
			more4          = inputImage[imore4];
		}
	}

	float G1_r = 0;
	float G2_r = 0;
	float G1_g = 0;
	float G2_g = 0;
	float G1_b = 0;
	float G2_b = 0;

	G1_r = abs(static_cast<float>(square_middle1.x) - static_cast<float>(square_left1.x))   +
		   abs(static_cast<float>(square_right1.x)  - static_cast<float>(square_middle1.x)) +
		   abs(static_cast<float>(square_middle2.x) - static_cast<float>(square_left2.x))   +
		   abs(static_cast<float>(square_right2.x)  - static_cast<float>(square_middle2.x)) +
		   abs(static_cast<float>(more3.x)          - static_cast<float>(circle_right2.x))  +
		   abs(static_cast<float>(circle_left2.x)   - static_cast<float>(more4.x))          +
		   abs(static_cast<float>(circle_right1.x)  - static_cast<float>(circle_left1.x))   +
		   abs(static_cast<float>(circle_right2.x)  - static_cast<float>(circle_left2.x))   +
		   abs(static_cast<float>(circle_right3.x)  - static_cast<float>(circle_left3.x));

	G2_r = abs(static_cast<float>(square_left1.x)   - static_cast<float>(square_left2.x))   +
		   abs(static_cast<float>(square_middle1.x) - static_cast<float>(square_middle2.x)) +
		   abs(static_cast<float>(square_right2.x)  - static_cast<float>(square_right1.x))  +
		   abs(static_cast<float>(more1.x)          - static_cast<float>(square_middle1.x)) +
		   abs(static_cast<float>(square_middle2.x) - static_cast<float>(more2.x))          +
		   abs(static_cast<float>(circle_left1.x)   - static_cast<float>(circle_left2.x))   +
		   abs(static_cast<float>(circle_left2.x)   - static_cast<float>(circle_left3.x))   +
		   abs(static_cast<float>(circle_right1.x)  - static_cast<float>(circle_right2.x))  +
		   abs(static_cast<float>(circle_right2.x)  - static_cast<float>(circle_right3.x));

	G1_g = abs(static_cast<float>(square_middle1.y) - static_cast<float>(square_left1.y))   +
		   abs(static_cast<float>(square_right1.y)  - static_cast<float>(square_middle1.y)) +
		   abs(static_cast<float>(square_middle2.y) - static_cast<float>(square_left2.y))   +
		   abs(static_cast<float>(square_right2.y)  - static_cast<float>(square_middle2.y)) +
		   abs(static_cast<float>(more3.y)          - static_cast<float>(circle_right2.y))  +
		   abs(static_cast<float>(circle_left2.y)   - static_cast<float>(more4.y))          +
		   abs(static_cast<float>(circle_right1.y)  - static_cast<float>(circle_left1.y))   +
		   abs(static_cast<float>(circle_right2.y)  - static_cast<float>(circle_left2.y))   +
		   abs(static_cast<float>(circle_right3.y)  - static_cast<float>(circle_left3.y));

	G2_g = abs(static_cast<float>(square_left1.y)   - static_cast<float>(square_left2.y))   +
		   abs(static_cast<float>(square_middle1.y) - static_cast<float>(square_middle2.y)) +
		   abs(static_cast<float>(square_right2.y)  - static_cast<float>(square_right1.y))  +
		   abs(static_cast<float>(more1.y)          - static_cast<float>(square_middle1.y)) +
		   abs(static_cast<float>(square_middle2.y) - static_cast<float>(more2.y))          +
		   abs(static_cast<float>(circle_left1.y)   - static_cast<float>(circle_left2.y))   +
		   abs(static_cast<float>(circle_left2.y)   - static_cast<float>(circle_left3.y))   +
		   abs(static_cast<float>(circle_right1.y)  - static_cast<float>(circle_right2.y))  +
		   abs(static_cast<float>(circle_right2.y)  - static_cast<float>(circle_right3.y));

	G1_b = abs(static_cast<float>(square_middle1.z) - static_cast<float>(square_left1.z))   +
		   abs(static_cast<float>(square_right1.z)  - static_cast<float>(square_middle1.z)) +
		   abs(static_cast<float>(square_middle2.z) - static_cast<float>(square_left2.z))   +
		   abs(static_cast<float>(square_right2.z)  - static_cast<float>(square_middle2.z)) +
		   abs(static_cast<float>(more3.z)          - static_cast<float>(circle_right2.z))  +
		   abs(static_cast<float>(circle_left2.z)   - static_cast<float>(more4.z))          +
		   abs(static_cast<float>(circle_right1.z)  - static_cast<float>(circle_left1.z))   +
		   abs(static_cast<float>(circle_right2.z)  - static_cast<float>(circle_left2.z))   +
		   abs(static_cast<float>(circle_right3.z)  - static_cast<float>(circle_left3.z));

	G2_b = abs(static_cast<float>(square_left1.z)   - static_cast<float>(square_left2.z))   +
		   abs(static_cast<float>(square_middle1.z) - static_cast<float>(square_middle2.z)) +
		   abs(static_cast<float>(square_right2.z)  - static_cast<float>(square_right1.z))  +
		   abs(static_cast<float>(more1.z)          - static_cast<float>(square_middle1.z)) +
		   abs(static_cast<float>(square_middle2.z) - static_cast<float>(more2.z))          +
		   abs(static_cast<float>(circle_left1.z)   - static_cast<float>(circle_left2.z))   +
		   abs(static_cast<float>(circle_left2.z)   - static_cast<float>(circle_left3.z))   +
		   abs(static_cast<float>(circle_right1.z)  - static_cast<float>(circle_right2.z))  +
		   abs(static_cast<float>(circle_right2.z)  - static_cast<float>(circle_right3.z));

	float output_b;
	float output_g;
	float output_r;

	//b
	if (100 * (1 + G1_r) > 100 * threshold*(1 + G2_r))
	{
		float val = attract(static_cast<float>(more1.x), static_cast<float>(square_middle1.x), static_cast<float>(square_middle2.x), static_cast<float>(more2.x), 1.);
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_b = val;
	}
	else if (100 * (1 + G2_r) > 100 * threshold*(1 + G1_r))
	{
		float val = attract(static_cast<float>(more4.x), static_cast<float>(circle_left2.x), static_cast<float>(circle_right2.x), static_cast<float>(more3.x), 1.);
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_b = val;
	}
	else
	{
		float w1 = 1 / (1 + pow(G1_r, k));
		float w2 = 1 / (1 + pow(G2_r, k));
		float weight1 = w1 / (w1 + w2);
		float weight2 = w2 / (w1 + w2);
		float p1 = attract(static_cast<float>(more1.x), static_cast<float>(square_middle1.x), static_cast<float>(square_middle2.x), static_cast<float>(more2.x), 1. / 6.);
		float p2 = attract(static_cast<float>(more4.x), static_cast<float>(circle_left2.x), static_cast<float>(circle_right2.x), static_cast<float>(more3.x), 1. / 6.);
		float val = weight1 * p1 + weight2 * p2;
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_b = val;
	}

	//g
	if (100 * (1 + G1_g) > 100 * threshold*(1 + G2_g))
	{
		float val = attract(static_cast<float>(more1.y), static_cast<float>(square_middle1.y), static_cast<float>(square_middle2.y), static_cast<float>(more2.y), 1.);
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_g = val;
	}
	else if (100 * (1 + G2_g) > 100 * threshold*(1 + G1_g))
	{
		float val = attract(static_cast<float>(more4.y), static_cast<float>(circle_left2.y), static_cast<float>(circle_right2.y), static_cast<float>(more3.y), 1.);
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_g = val;
	}
	else
	{
		float w1 = 1 / (1 + pow(G1_g, k));
		float w2 = 1 / (1 + pow(G2_g, k));
		float weight1 = w1 / (w1 + w2);
		float weight2 = w2 / (w1 + w2);
		float p1 = attract(static_cast<float>(more1.y), static_cast<float>(square_middle1.y), static_cast<float>(square_middle2.y), static_cast<float>(more2.y), 1. / 6.);
		float p2 = attract(static_cast<float>(more4.y), static_cast<float>(circle_left2.y), static_cast<float>(circle_right2.y), static_cast<float>(more3.y), 1. / 6.);
		float val = weight1 * p1 + weight2 * p2;
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_g = val;
	}

	//r
	if (100 * (1 + G1_b) > 100 * threshold*(1 + G2_b))
	{
		float val = attract(static_cast<float>(more1.z), static_cast<float>(square_middle1.z), static_cast<float>(square_middle2.z), static_cast<float>(more2.z), 1.);
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_r = val;
	}
	else if (100 * (1 + G2_b) > 100 * threshold*(1 + G1_b))
	{
		float val = attract(static_cast<float>(more4.z), static_cast<float>(circle_left2.z), static_cast<float>(circle_right2.z), static_cast<float>(more3.z), 1.);
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_r = val;
	}
	else
	{
		float w1 = 1 / (1 + pow(G1_b, k));
		float w2 = 1 / (1 + pow(G2_b, k));
		float weight1 = w1 / (w1 + w2);
		float weight2 = w2 / (w1 + w2);
		float p1 = attract(static_cast<float>(more1.z), static_cast<float>(square_middle1.z), static_cast<float>(square_middle2.z), static_cast<float>(more2.z), 1. / 6.);
		float p2 = attract(static_cast<float>(more4.z), static_cast<float>(circle_left2.z), static_cast<float>(circle_right2.z), static_cast<float>(more3.z), 1. / 6.);
		float val = weight1 * p1 + weight2 * p2;
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_r = val;
	}

	outputImage[out].x = output_b;
	outputImage[out].y = output_g;
	outputImage[out].z = output_r;
}

__global__ void secondpass2(const uchar3* const inputImage,
	                        uchar3* const outputImage,
	                        int numRows,
	                        int numCols)
{
	double threshold = 1.15;
	int k = 5;

	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	if (col >= numCols || row > numRows)
		return;

	
	int out = row * numCols + col;

	if (row % 2 == 0)
	{
		if (col % 2 == 0)
		{
			outputImage[out] = inputImage[out];
			return;
		}
	}
	else
	{
		outputImage[out] = inputImage[out];
		return;
	}

	int x = row;
	int y = col;

	int isquare_left1 = (x - 1) * numCols + (y - 2);
	int isquare_left2 = (x + 1) * numCols + (y - 2);
	int icircle_left1 = (x - 2) * numCols + (y - 1);
	int icircle_left2 = (x)  * numCols + (y - 1);
	int icircle_left3 = (x + 2) * numCols + (y - 1);
	int isquare_middle1 = (x - 1) * numCols + (y);
	int isquare_middle2 = (x + 1) * numCols + (y);
	int icircle_right1 = (x - 2)  * numCols + (y + 1);
	int icircle_right2 = (x) * numCols + (y + 1);
	int icircle_right3 = (x + 2) * numCols + (y + 1);
	int isquare_right1 = (x - 1) * numCols + (y + 2);
	int isquare_right2 = (x + 1) * numCols + (y + 2);
	int imore1 = (x - 3) * numCols + (y);
	int imore2 = (x + 3) * numCols + (y);
	int imore3 = (x) * numCols + (y + 3);
	int imore4 = (x) * numCols + (y - 3);

	uchar3 square_left1;
	uchar3 square_left2;
	uchar3 circle_left1;
	uchar3 circle_left2;
	uchar3 circle_left3;
	uchar3 square_middle1;
	uchar3 square_middle2;
	uchar3 circle_right1;
	uchar3 circle_right2;
	uchar3 circle_right3;
	uchar3 square_right1;
	uchar3 square_right2;
	uchar3 more1;
	uchar3 more2;
	uchar3 more3;
	uchar3 more4;

	if (x == 0)
	{
		if (y == 0)
		{
			square_left1 = make_uchar3(0, 0, 0);
			square_left2 = make_uchar3(0, 0, 0);
			circle_left1 = make_uchar3(0, 0, 0);
			circle_left2 = make_uchar3(0, 0, 0);
			circle_left3 = make_uchar3(0, 0, 0);
			square_middle1 = make_uchar3(0, 0, 0);
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = make_uchar3(0, 0, 0);
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = make_uchar3(0, 0, 0);
			square_right2 = inputImage[isquare_right2];
			more1 = make_uchar3(0, 0, 0);
			more2 = inputImage[imore2];
			more3 = inputImage[imore3];
			more4 = make_uchar3(0, 0, 0);
		}
		else if (y == 1)
		{
			square_left1 = make_uchar3(0, 0, 0);
			square_left2 = make_uchar3(0, 0, 0);
			circle_left1 = make_uchar3(0, 0, 0);
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = make_uchar3(0, 0, 0);
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = make_uchar3(0, 0, 0);
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = make_uchar3(0, 0, 0);
			square_right2 = inputImage[isquare_right2];
			more1 = make_uchar3(0, 0, 0);
			more2 = inputImage[imore2];
			more3 = inputImage[imore3];
			more4 = make_uchar3(0, 0, 0);
		}
		else if (y == 2)
		{
			square_left1 = make_uchar3(0, 0, 0);
			square_left2 = inputImage[isquare_left2];
			circle_left1 = make_uchar3(0, 0, 0);
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = make_uchar3(0, 0, 0);
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = make_uchar3(0, 0, 0);
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = make_uchar3(0, 0, 0);
			square_right2 = inputImage[isquare_right2];
			more1 = make_uchar3(0, 0, 0);
			more2 = inputImage[imore2];
			more3 = inputImage[imore3];
			more4 = make_uchar3(0, 0, 0);
		}
		else if (y == numCols - 3)
		{
			square_left1 = make_uchar3(0, 0, 0);
			square_left2 = inputImage[isquare_left2];
			circle_left1 = make_uchar3(0, 0, 0);
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = make_uchar3(0, 0, 0);
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = make_uchar3(0, 0, 0);
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = make_uchar3(0, 0, 0);
			square_right2 = inputImage[isquare_right2];
			more1 = make_uchar3(0, 0, 0);
			more2 = inputImage[imore2];
			more3 = make_uchar3(0, 0, 0);
			more4 = inputImage[imore4];
		}
		else if (y == numCols - 2)
		{
			square_left1 = make_uchar3(0, 0, 0);
			square_left2 = inputImage[isquare_left2];
			circle_left1 = make_uchar3(0, 0, 0);
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = make_uchar3(0, 0, 0);
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = make_uchar3(0, 0, 0);
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = make_uchar3(0, 0, 0);
			square_right2 = make_uchar3(0, 0, 0);
			more1 = make_uchar3(0, 0, 0);
			more2 = inputImage[imore2];
			more3 = make_uchar3(0, 0, 0);
			more4 = inputImage[imore4];
		}
		else if (y == numCols - 1)
		{
			square_left1 = make_uchar3(0, 0, 0);
			square_left2 = inputImage[isquare_left2];
			circle_left1 = make_uchar3(0, 0, 0);
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = make_uchar3(0, 0, 0);
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = make_uchar3(0, 0, 0);
			circle_right2 = make_uchar3(0, 0, 0);
			circle_right3 = make_uchar3(0, 0, 0);
			square_right1 = make_uchar3(0, 0, 0);
			square_right2 = make_uchar3(0, 0, 0);
			more1 = make_uchar3(0, 0, 0);
			more2 = inputImage[imore2];
			more3 = make_uchar3(0, 0, 0);
			more4 = inputImage[imore4];
		}
		else
		{
			square_left1 = make_uchar3(0, 0, 0);
			square_left2 = inputImage[isquare_left2];
			circle_left1 = make_uchar3(0, 0, 0);
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = make_uchar3(0, 0, 0);
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = make_uchar3(0, 0, 0);
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = make_uchar3(0, 0, 0);
			square_right2 = inputImage[isquare_right2];
			more1 = make_uchar3(0, 0, 0);
			more2 = inputImage[imore2];
			more3 = inputImage[imore3];
			more4 = inputImage[imore4];
		}
	}
	else if (x == 1)
	{
		if (y == 0)
		{
			square_left1 = make_uchar3(0, 0, 0);
			square_left2 = make_uchar3(0, 0, 0);
			circle_left1 = make_uchar3(0, 0, 0);
			circle_left2 = make_uchar3(0, 0, 0);
			circle_left3 = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = make_uchar3(0, 0, 0);
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = make_uchar3(0, 0, 0);
			more2 = inputImage[imore2];
			more3 = inputImage[imore3];
			more4 = make_uchar3(0, 0, 0);
		}
		else if (y == 1)
		{
			square_left1 = make_uchar3(0, 0, 0);
			square_left2 = make_uchar3(0, 0, 0);
			circle_left1 = make_uchar3(0, 0, 0);
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = make_uchar3(0, 0, 0);
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = make_uchar3(0, 0, 0);
			more2 = inputImage[imore2];
			more3 = inputImage[imore3];
			more4 = make_uchar3(0, 0, 0);
		}
		else if (y == 2)
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = make_uchar3(0, 0, 0);
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = make_uchar3(0, 0, 0);
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = make_uchar3(0, 0, 0);
			more2 = inputImage[imore2];
			more3 = inputImage[imore3];
			more4 = make_uchar3(0, 0, 0);
		}
		else if (y == numCols - 3)
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = make_uchar3(0, 0, 0);
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = make_uchar3(0, 0, 0);
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = make_uchar3(0, 0, 0);
			more2 = inputImage[imore2];
			more3 = make_uchar3(0, 0, 0);
			more4 = inputImage[imore4];
		}
		else if (y == numCols - 2)
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = make_uchar3(0, 0, 0);
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = make_uchar3(0, 0, 0);
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = make_uchar3(0, 0, 0);
			square_right2 = make_uchar3(0, 0, 0);
			more1 = make_uchar3(0, 0, 0);
			more2 = inputImage[imore2];
			more3 = make_uchar3(0, 0, 0);
			more4 = inputImage[imore4];
		}
		else if (y == numCols - 1)
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = make_uchar3(0, 0, 0);
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = make_uchar3(0, 0, 0);
			circle_right2 = make_uchar3(0, 0, 0);
			circle_right3 = make_uchar3(0, 0, 0);
			square_right1 = make_uchar3(0, 0, 0);
			square_right2 = make_uchar3(0, 0, 0);
			more1 = make_uchar3(0, 0, 0);
			more2 = inputImage[imore2];
			more3 = make_uchar3(0, 0, 0);
			more4 = inputImage[imore4];
		}
		else
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = make_uchar3(0, 0, 0);
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = make_uchar3(0, 0, 0);
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = make_uchar3(0, 0, 0);
			more2 = inputImage[imore2];
			more3 = inputImage[imore3];
			more4 = inputImage[imore4];
		}
	}
	else if (x == 2)
	{
		if (y == 0)
		{
			square_left1 = make_uchar3(0, 0, 0);
			square_left2 = make_uchar3(0, 0, 0);
			circle_left1 = make_uchar3(0, 0, 0);
			circle_left2 = make_uchar3(0, 0, 0);
			circle_left3 = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = make_uchar3(0, 0, 0);
			more2 = inputImage[imore2];
			more3 = inputImage[imore3];
			more4 = make_uchar3(0, 0, 0);
		}
		else if (y == 1)
		{
			square_left1 = make_uchar3(0, 0, 0);
			square_left2 = make_uchar3(0, 0, 0);
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = make_uchar3(0, 0, 0);
			more2 = inputImage[imore2];
			more3 = inputImage[imore3];
			more4 = make_uchar3(0, 0, 0);
		}
		else if (y == 2)
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = make_uchar3(0, 0, 0);
			more2 = inputImage[imore2];
			more3 = inputImage[imore3];
			more4 = make_uchar3(0, 0, 0);
		}
		else if (y == numCols - 3)
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = make_uchar3(0, 0, 0);
			more2 = inputImage[imore2];
			more3 = make_uchar3(0, 0, 0);
			more4 = inputImage[imore4];
		}
		else if (y == numCols - 2)
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = make_uchar3(0, 0, 0);
			square_right2 = make_uchar3(0, 0, 0);
			more1 = make_uchar3(0, 0, 0);
			more2 = inputImage[imore2];
			more3 = make_uchar3(0, 0, 0);
			more4 = inputImage[imore4];
		}
		else if (y == numCols - 1)
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = make_uchar3(0, 0, 0);
			circle_right2 = make_uchar3(0, 0, 0);
			circle_right3 = make_uchar3(0, 0, 0);
			square_right1 = make_uchar3(0, 0, 0);
			square_right2 = make_uchar3(0, 0, 0);
			more1 = make_uchar3(0, 0, 0);
			more2 = inputImage[imore2];
			more3 = make_uchar3(0, 0, 0);
			more4 = inputImage[imore4];
		}
		else
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = make_uchar3(0, 0, 0);
			more2 = inputImage[imore2];
			more3 = inputImage[imore3];
			more4 = inputImage[imore4];
		}
	}
	else if (x == numRows - 3)
	{
		if (y == 0)
		{
			square_left1 = make_uchar3(0, 0, 0);
			square_left2 = make_uchar3(0, 0, 0);
			circle_left1 = make_uchar3(0, 0, 0);
			circle_left2 = make_uchar3(0, 0, 0);
			circle_left3 = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = inputImage[imore1];
			more2 = make_uchar3(0, 0, 0);
			more3 = inputImage[imore3];
			more4 = make_uchar3(0, 0, 0);
		}
		else if (y == 1)
		{
			square_left1 = make_uchar3(0, 0, 0);
			square_left2 = make_uchar3(0, 0, 0);
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = inputImage[imore1];
			more2 = make_uchar3(0, 0, 0);
			more3 = inputImage[imore3];
			more4 = make_uchar3(0, 0, 0);
		}
		else if (y == 2)
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = inputImage[imore1];
			more2 = make_uchar3(0, 0, 0);
			more3 = inputImage[imore3];
			more4 = make_uchar3(0, 0, 0);
		}
		else if (y == numCols - 3)
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = inputImage[imore1];
			more2 = make_uchar3(0, 0, 0);
			more3 = make_uchar3(0, 0, 0);
			more4 = inputImage[imore4];
		}
		else if (y == numCols - 2)
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = make_uchar3(0, 0, 0);
			square_right2 = make_uchar3(0, 0, 0);
			more1 = inputImage[imore1];
			more2 = make_uchar3(0, 0, 0);
			more3 = make_uchar3(0, 0, 0);
			more4 = inputImage[imore4];
		}
		else if (y == numCols - 1)
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = make_uchar3(0, 0, 0);
			circle_right2 = make_uchar3(0, 0, 0);
			circle_right3 = make_uchar3(0, 0, 0);
			square_right1 = make_uchar3(0, 0, 0);
			square_right2 = make_uchar3(0, 0, 0);
			more1 = inputImage[imore1];
			more2 = make_uchar3(0, 0, 0);
			more3 = make_uchar3(0, 0, 0);
			more4 = inputImage[imore4];
		}
		else
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = inputImage[imore1];
			more2 = make_uchar3(0, 0, 0);
			more3 = inputImage[imore3];
			more4 = inputImage[imore4];
		}
	}
	else if (x == numRows - 2)
	{
		if (y == 0)
		{
			square_left1 = make_uchar3(0, 0, 0);
			square_left2 = make_uchar3(0, 0, 0);
			circle_left1 = make_uchar3(0, 0, 0);
			circle_left2 = make_uchar3(0, 0, 0);
			circle_left3 = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = make_uchar3(0, 0, 0);
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = inputImage[imore1];
			more2 = make_uchar3(0, 0, 0);
			more3 = inputImage[imore3];
			more4 = make_uchar3(0, 0, 0);
		}
		else if (y == 1)
		{
			square_left1 = make_uchar3(0, 0, 0);
			square_left2 = make_uchar3(0, 0, 0);
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = make_uchar3(0, 0, 0);
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = inputImage[imore1];
			more2 = make_uchar3(0, 0, 0);
			more3 = inputImage[imore3];
			more4 = make_uchar3(0, 0, 0);
		}
		else if (y == 2)
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = make_uchar3(0, 0, 0);
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = inputImage[imore1];
			more2 = make_uchar3(0, 0, 0);
			more3 = inputImage[imore3];
			more4 = make_uchar3(0, 0, 0);
		}
		else if (y == numCols - 3)
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = make_uchar3(0, 0, 0);
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = inputImage[imore1];
			more2 = make_uchar3(0, 0, 0);
			more3 = make_uchar3(0, 0, 0);
			more4 = inputImage[imore4];
		}
		else if (y == numCols - 2)
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = make_uchar3(0, 0, 0);
			square_right1 = make_uchar3(0, 0, 0);
			square_right2 = make_uchar3(0, 0, 0);
			more1 = inputImage[imore1];
			more2 = make_uchar3(0, 0, 0);
			more3 = make_uchar3(0, 0, 0);
			more4 = inputImage[imore4];
		}
		else if (y == numCols - 1)
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = make_uchar3(0, 0, 0);
			circle_right2 = make_uchar3(0, 0, 0);
			circle_right3 = make_uchar3(0, 0, 0);
			square_right1 = make_uchar3(0, 0, 0);
			square_right2 = make_uchar3(0, 0, 0);
			more1 = inputImage[imore1];
			more2 = make_uchar3(0, 0, 0);
			more3 = make_uchar3(0, 0, 0);
			more4 = inputImage[imore4];
		}
		else
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = make_uchar3(0, 0, 0);
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = inputImage[imore1];
			more2 = make_uchar3(0, 0, 0);
			more3 = inputImage[imore3];
			more4 = inputImage[imore4];
		}
	}
	else if (x == numRows - 1)
	{
		if (y == 0)
		{
			square_left1 = make_uchar3(0, 0, 0);
			square_left2 = make_uchar3(0, 0, 0);
			circle_left1 = make_uchar3(0, 0, 0);
			circle_left2 = make_uchar3(0, 0, 0);
			circle_left3 = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = make_uchar3(0, 0, 0);
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = make_uchar3(0, 0, 0);
			square_right1 = inputImage[isquare_right1];
			square_right2 = make_uchar3(0, 0, 0);
			more1 = inputImage[imore1];
			more2 = make_uchar3(0, 0, 0);
			more3 = inputImage[imore3];
			more4 = make_uchar3(0, 0, 0);
		}
		else if (y == 1)
		{
			square_left1 = make_uchar3(0, 0, 0);
			square_left2 = make_uchar3(0, 0, 0);
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = make_uchar3(0, 0, 0);
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = make_uchar3(0, 0, 0);
			square_right1 = inputImage[isquare_right1];
			square_right2 = make_uchar3(0, 0, 0);
			more1 = inputImage[imore1];
			more2 = make_uchar3(0, 0, 0);
			more3 = inputImage[imore3];
			more4 = make_uchar3(0, 0, 0);
		}
		else if (y == 2)
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = make_uchar3(0, 0, 0);
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = make_uchar3(0, 0, 0);
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = make_uchar3(0, 0, 0);
			square_right1 = inputImage[isquare_right1];
			square_right2 = make_uchar3(0, 0, 0);
			more1 = inputImage[imore1];
			more2 = make_uchar3(0, 0, 0);
			more3 = inputImage[imore3];
			more4 = make_uchar3(0, 0, 0);
		}
		else if (y == numCols - 3)
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = make_uchar3(0, 0, 0);
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = make_uchar3(0, 0, 0);
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = make_uchar3(0, 0, 0);
			square_right1 = inputImage[isquare_right1];
			square_right2 = make_uchar3(0, 0, 0);
			more1 = inputImage[imore1];
			more2 = make_uchar3(0, 0, 0);
			more3 = make_uchar3(0, 0, 0);
			more4 = inputImage[imore4];
		}
		else if (y == numCols - 2)
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = make_uchar3(0, 0, 0);
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = make_uchar3(0, 0, 0);
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = make_uchar3(0, 0, 0);
			square_right1 = make_uchar3(0, 0, 0);
			square_right2 = make_uchar3(0, 0, 0);
			more1 = inputImage[imore1];
			more2 = make_uchar3(0, 0, 0);
			more3 = make_uchar3(0, 0, 0);
			more4 = inputImage[imore4];
		}
		else if (y == numCols - 1)
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = make_uchar3(0, 0, 0);
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = make_uchar3(0, 0, 0);
			circle_right1 = make_uchar3(0, 0, 0);
			circle_right2 = make_uchar3(0, 0, 0);
			circle_right3 = make_uchar3(0, 0, 0);
			square_right1 = make_uchar3(0, 0, 0);
			square_right2 = make_uchar3(0, 0, 0);
			more1 = inputImage[imore1];
			more2 = make_uchar3(0, 0, 0);
			more3 = make_uchar3(0, 0, 0);
			more4 = inputImage[imore4];
		}
		else
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = make_uchar3(0, 0, 0);
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = make_uchar3(0, 0, 0);
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = make_uchar3(0, 0, 0);
			square_right1 = inputImage[isquare_right1];
			square_right2 = make_uchar3(0, 0, 0);
			more1 = inputImage[imore1];
			more2 = make_uchar3(0, 0, 0);
			more3 = inputImage[imore3];
			more4 = inputImage[imore4];
		}
	}
	else
	{
		if (y == 0)
		{
			square_left1 = make_uchar3(0, 0, 0);
			square_left2 = make_uchar3(0, 0, 0);
			circle_left1 = make_uchar3(0, 0, 0);
			circle_left2 = make_uchar3(0, 0, 0);
			circle_left3 = make_uchar3(0, 0, 0);
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = inputImage[imore1];
			more2 = inputImage[imore2];
			more3 = inputImage[imore3];
			more4 = make_uchar3(0, 0, 0);
		}
		else if (y == 1)
		{
			square_left1 = make_uchar3(0, 0, 0);
			square_left2 = make_uchar3(0, 0, 0);
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = inputImage[imore1];
			more2 = inputImage[imore2];
			more3 = inputImage[imore3];
			more4 = make_uchar3(0, 0, 0);
		}
		else if (y == 2)
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = inputImage[imore1];
			more2 = inputImage[imore2];
			more3 = inputImage[imore3];
			more4 = make_uchar3(0, 0, 0);
		}
		else if (y == numCols - 3)
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = inputImage[imore1];
			more2 = inputImage[imore2];
			more3 = make_uchar3(0, 0, 0);
			more4 = inputImage[imore4];
		}
		else if (y == numCols - 2)
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = make_uchar3(0, 0, 0);
			square_right2 = make_uchar3(0, 0, 0);
			more1 = inputImage[imore1];
			more2 = inputImage[imore2];
			more3 = make_uchar3(0, 0, 0);
			more4 = inputImage[imore4];
		}
		else if (y == numCols - 1)
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = make_uchar3(0, 0, 0);
			circle_right2 = make_uchar3(0, 0, 0);
			circle_right3 = make_uchar3(0, 0, 0);
			square_right1 = make_uchar3(0, 0, 0);
			square_right2 = make_uchar3(0, 0, 0);
			more1 = inputImage[imore1];
			more2 = inputImage[imore2];
			more3 = make_uchar3(0, 0, 0);
			more4 = inputImage[imore4];
		}
		else
		{
			square_left1 = inputImage[isquare_left1];
			square_left2 = inputImage[isquare_left2];
			circle_left1 = inputImage[icircle_left1];
			circle_left2 = inputImage[icircle_left2];
			circle_left3 = inputImage[icircle_left3];
			square_middle1 = inputImage[isquare_middle1];
			square_middle2 = inputImage[isquare_middle2];
			circle_right1 = inputImage[icircle_right1];
			circle_right2 = inputImage[icircle_right2];
			circle_right3 = inputImage[icircle_right3];
			square_right1 = inputImage[isquare_right1];
			square_right2 = inputImage[isquare_right2];
			more1 = inputImage[imore1];
			more2 = inputImage[imore2];
			more3 = inputImage[imore3];
			more4 = inputImage[imore4];
		}
	}

	float G1_r = 0;
	float G2_r = 0;
	float G1_g = 0;
	float G2_g = 0;
	float G1_b = 0;
	float G2_b = 0;

	G1_r = abs(static_cast<float>(square_middle1.x) - static_cast<float>(square_left1.x))   +
		   abs(static_cast<float>(square_right1.x)  - static_cast<float>(square_middle1.x)) +
		   abs(static_cast<float>(square_middle2.x) - static_cast<float>(square_left2.x))   +
		   abs(static_cast<float>(square_right2.x)  - static_cast<float>(square_middle2.x)) +
		   abs(static_cast<float>(more3.x)          - static_cast<float>(circle_right2.x))  +
		   abs(static_cast<float>(circle_left2.x)   - static_cast<float>(more4.x))          +
		   abs(static_cast<float>(circle_right1.x)  - static_cast<float>(circle_left1.x))   +
		   abs(static_cast<float>(circle_right2.x)  - static_cast<float>(circle_left2.x))   +
		   abs(static_cast<float>(circle_right3.x)  - static_cast<float>(circle_left3.x));

	G2_r = abs(static_cast<float>(square_left1.x)   - static_cast<float>(square_left2.x))   +
		   abs(static_cast<float>(square_middle1.x) - static_cast<float>(square_middle2.x)) +
		   abs(static_cast<float>(square_right2.x)  - static_cast<float>(square_right1.x))  +
		   abs(static_cast<float>(more1.x)          - static_cast<float>(square_middle1.x)) +
		   abs(static_cast<float>(square_middle2.x) - static_cast<float>(more2.x))          +
		   abs(static_cast<float>(circle_left1.x)   - static_cast<float>(circle_left2.x))   +
		   abs(static_cast<float>(circle_left2.x)   - static_cast<float>(circle_left3.x))   +
		   abs(static_cast<float>(circle_right1.x)  - static_cast<float>(circle_right2.x))  +
		   abs(static_cast<float>(circle_right2.x)  - static_cast<float>(circle_right3.x));

	G1_g = abs(static_cast<float>(square_middle1.y) - static_cast<float>(square_left1.y))   +
		   abs(static_cast<float>(square_right1.y)  - static_cast<float>(square_middle1.y)) +
		   abs(static_cast<float>(square_middle2.y) - static_cast<float>(square_left2.y))   +
		   abs(static_cast<float>(square_right2.y)  - static_cast<float>(square_middle2.y)) +
		   abs(static_cast<float>(more3.y)          - static_cast<float>(circle_right2.y))  +
		   abs(static_cast<float>(circle_left2.y)   - static_cast<float>(more4.y))          +
		   abs(static_cast<float>(circle_right1.y)  - static_cast<float>(circle_left1.y))   +
		   abs(static_cast<float>(circle_right2.y)  - static_cast<float>(circle_left2.y))   +
		   abs(static_cast<float>(circle_right3.y)  - static_cast<float>(circle_left3.y));

	G2_g = abs(static_cast<float>(square_left1.y)   - static_cast<float>(square_left2.y))   +
		   abs(static_cast<float>(square_middle1.y) - static_cast<float>(square_middle2.y)) +
		   abs(static_cast<float>(square_right2.y)  - static_cast<float>(square_right1.y))  +
		   abs(static_cast<float>(more1.y)          - static_cast<float>(square_middle1.y)) +
		   abs(static_cast<float>(square_middle2.y) - static_cast<float>(more2.y))          +
		   abs(static_cast<float>(circle_left1.y)   - static_cast<float>(circle_left2.y))   +
		   abs(static_cast<float>(circle_left2.y)   - static_cast<float>(circle_left3.y))   +
		   abs(static_cast<float>(circle_right1.y)  - static_cast<float>(circle_right2.y))  +
		   abs(static_cast<float>(circle_right2.y)  - static_cast<float>(circle_right3.y));

	G1_b = abs(static_cast<float>(square_middle1.z) - static_cast<float>(square_left1.z))   +
		   abs(static_cast<float>(square_right1.z)  - static_cast<float>(square_middle1.z)) +
		   abs(static_cast<float>(square_middle2.z) - static_cast<float>(square_left2.z))   +
		   abs(static_cast<float>(square_right2.z)  - static_cast<float>(square_middle2.z)) +
		   abs(static_cast<float>(more3.z)          - static_cast<float>(circle_right2.z))  +
		   abs(static_cast<float>(circle_left2.z)   - static_cast<float>(more4.z))          +
		   abs(static_cast<float>(circle_right1.z)  - static_cast<float>(circle_left1.z))   +
		   abs(static_cast<float>(circle_right2.z)  - static_cast<float>(circle_left2.z))   +
		   abs(static_cast<float>(circle_right3.z)  - static_cast<float>(circle_left3.z));

	G2_b = abs(static_cast<float>(square_left1.z)   - static_cast<float>(square_left2.z))   +
		   abs(static_cast<float>(square_middle1.z) - static_cast<float>(square_middle2.z)) +
		   abs(static_cast<float>(square_right2.z)  - static_cast<float>(square_right1.z))  +
		   abs(static_cast<float>(more1.z)          - static_cast<float>(square_middle1.z)) +
		   abs(static_cast<float>(square_middle2.z) - static_cast<float>(more2.z))          +
		   abs(static_cast<float>(circle_left1.z)   - static_cast<float>(circle_left2.z))   +
		   abs(static_cast<float>(circle_left2.z)   - static_cast<float>(circle_left3.z))   +
		   abs(static_cast<float>(circle_right1.z)  - static_cast<float>(circle_right2.z))  +
		   abs(static_cast<float>(circle_right2.z)  - static_cast<float>(circle_right3.z));

	float output_b;
	float output_g;
	float output_r;

	//b
	if (100 * (1 + G1_r) > 100 * threshold*(1 + G2_r))
	{
		float val = attract(static_cast<float>(more1.x), static_cast<float>(square_middle1.x), static_cast<float>(square_middle2.x), static_cast<float>(more2.x), 1.);
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_b = val;
	}
	else if (100 * (1 + G2_r) > 100 * threshold*(1 + G1_r))
	{
		float val = attract(static_cast<float>(more4.x), static_cast<float>(circle_left2.x), static_cast<float>(circle_right2.x), static_cast<float>(more3.x), 1.);
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_b = val;
	}
	else
	{
		float w1 = 1 / (1 + pow(G1_r, k));
		float w2 = 1 / (1 + pow(G2_r, k));
		float weight1 = w1 / (w1 + w2);
		float weight2 = w2 / (w1 + w2);
		float p1 = attract(static_cast<float>(more1.x), static_cast<float>(square_middle1.x), static_cast<float>(square_middle2.x), static_cast<float>(more2.x), 1. / 6.);
		float p2 = attract(static_cast<float>(more4.x), static_cast<float>(circle_left2.x), static_cast<float>(circle_right2.x), static_cast<float>(more3.x), 1. / 6.);
		float val = weight1 * p1 + weight2 * p2;
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_b = val;
	}

	//g
	if (100 * (1 + G1_g) > 100 * threshold*(1 + G2_g))
	{
		float val = attract(static_cast<float>(more1.y), static_cast<float>(square_middle1.y), static_cast<float>(square_middle2.y), static_cast<float>(more2.y), 1.);
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_g = val;
	}
	else if (100 * (1 + G2_g) > 100 * threshold*(1 + G1_g))
	{
		float val = attract(static_cast<float>(more4.y), static_cast<float>(circle_left2.y), static_cast<float>(circle_right2.y), static_cast<float>(more3.y), 1.);
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_g = val;
	}
	else
	{
		float w1 = 1 / (1 + pow(G1_g, k));
		float w2 = 1 / (1 + pow(G2_g, k));
		float weight1 = w1 / (w1 + w2);
		float weight2 = w2 / (w1 + w2);
		float p1 = attract(static_cast<float>(more1.y), static_cast<float>(square_middle1.y), static_cast<float>(square_middle2.y), static_cast<float>(more2.y), 1. / 6.);
		float p2 = attract(static_cast<float>(more4.y), static_cast<float>(circle_left2.y), static_cast<float>(circle_right2.y), static_cast<float>(more3.y), 1. / 6.);
		float val = weight1 * p1 + weight2 * p2;
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_g = val;
	}

	//r
	if (100 * (1 + G1_b) > 100 * threshold*(1 + G2_b))
	{
		float val = attract(static_cast<float>(more1.z), static_cast<float>(square_middle1.z), static_cast<float>(square_middle2.z), static_cast<float>(more2.z), 1.);
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_r = val;
	}
	else if (100 * (1 + G2_b) > 100 * threshold*(1 + G1_b))
	{
		float val = attract(static_cast<float>(more4.z), static_cast<float>(circle_left2.z), static_cast<float>(circle_right2.z), static_cast<float>(more3.z), 1.);
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_r = val;
	}
	else
	{
		float w1 = 1 / (1 + pow(G1_b, k));
		float w2 = 1 / (1 + pow(G2_b, k));
		float weight1 = w1 / (w1 + w2);
		float weight2 = w2 / (w1 + w2);
		float p1 = attract(static_cast<float>(more1.z), static_cast<float>(square_middle1.z), static_cast<float>(square_middle2.z), static_cast<float>(more2.z), 1. / 6.);
		float p2 = attract(static_cast<float>(more4.z), static_cast<float>(circle_left2.z), static_cast<float>(circle_right2.z), static_cast<float>(more3.z), 1. / 6.);
		float val = weight1 * p1 + weight2 * p2;
		if (val < 0)
			val = 0;
		if (val > 255)
			val = 255;
		output_r = val;
	}

	outputImage[out].x = output_b;
	outputImage[out].y = output_g;
	outputImage[out].z = output_r;
}

uchar3* dcci(uchar3* d_in, size_t numRows, size_t numCols)
{
	const dim3 blockSize(16, 16, 1);
	int a = numCols / blockSize.x;
	int b = numRows / blockSize.y;
	const dim3 gridSize(a + 1, b + 1, 1);
	const size_t numPixels = numRows * numCols;

	uchar3* d_out1;
	cudaMalloc((void**)&d_out1, sizeof(uchar3)*numPixels);
	cudaMemset(d_out1, 0, numPixels * sizeof(uchar3));

	uchar3* d_out2;
	cudaMalloc((void**)&d_out2, sizeof(uchar3)*numPixels);
	cudaMemset(d_out2, 0, numPixels * sizeof(uchar3));

	uchar3* d_out;
	cudaMalloc((void**)&d_out, sizeof(uchar3)*numPixels);
	cudaMemset(d_out, 0, numPixels * sizeof(uchar3));

	d_inputImageRGBA__ = d_in;
	d_outputImageRGBA1__ = d_out1;
	d_outputImageRGBA2__ = d_out2;
	d_outputImageRGBA__ = d_out;

	//first pass - white square
	firstpass <<<gridSize, blockSize>>> (d_in, d_out1, numRows, numCols);
	cudaDeviceSynchronize();
	//second pass - grey circles
	secondpass1 <<<gridSize, blockSize>>> (d_out1, d_out2, numRows, numCols);
	cudaDeviceSynchronize();
	//second pass - white circles
	secondpass2 <<<gridSize, blockSize>>> (d_out2, d_out, numRows, numCols);
	cudaDeviceSynchronize();

	//output
	uchar3* h_out;
	h_out = (uchar3*)malloc(sizeof(uchar3)*numPixels);
	cudaMemcpy(h_out, d_out, sizeof(uchar3)*numPixels, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(d_inputImageRGBA__);
	cudaFree(d_outputImageRGBA1__);
	cudaFree(d_outputImageRGBA2__);
	cudaFree(d_outputImageRGBA__);

	return h_out;
}



