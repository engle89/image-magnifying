#include <algorithm>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <stdlib.h>
#include <stdio.h>
//#include <math.h>


using namespace cv;
using namespace std;

float attract(float x, float y, float z, float t, float expose)
{
	expose = 1. / 2.;
	return pow((-2 * pow(x, expose) + 10 * pow(y, expose) + 10 * pow(z, expose) - 2 * pow(t, expose)) / 16., 1. / expose);
}

Mat interpolate_rgb(Mat X)
{
	int ws = 4; //window size
	double threshold = 1.15;
	int k = 5;
	Mat X_mirrors;
	Mat R_mirrors;
	cout << "test1" << endl;
	//this is for the pixel at border
	copyMakeBorder(X, X_mirrors, ws - 1, ws - 1, ws - 1, ws - 1, BORDER_REFLECT);
	Mat Result(2 * X.rows, 2 * X.cols, CV_8UC3, Scalar(0, 0, 0));

	//remember to correct offset because you extend the border before
	for (int i = ws - 1; i < X_mirrors.rows - ws + 1; i++) {
		for (int j = ws - 1; j < X_mirrors.cols - ws + 1; j++) {
			int ii = 2 * (i - ws + 1);
			int jj = 2 * (j - ws + 1);
			Vec3b pixel = X_mirrors.at<Vec3b>(i, j);
			Result.at<Vec3b>(ii, jj) = pixel;
		}
	}

	//fill white square
	for (int i = ws - 1; i < X_mirrors.rows - ws + 1; i++) {
		for (int j = ws - 1; j < X_mirrors.cols - ws + 1; j++) {
			int ii = 2 * (i - ws + 1) + 1;
			int jj = 2 * (j - ws + 1) + 1;

			Vec3b value;

			for (int col = 0; col < 3; col++)
			{
				double G1 = 0;
				double G2 = 0;
				for (int x = -1; x < 2; x++) {
					for (int y = -1; y < 2; y++) {
						//G1:
						Vec3b pix11 = X_mirrors.at<Vec3b>(i + x + 1, j + y);
						Vec3b pix12 = X_mirrors.at<Vec3b>(i + x, j + y + 1);
						G1 += abs(double(pix11[col]) - double(pix12[col]));
						//G2:
						Vec3b pix21 = X_mirrors.at<Vec3b>(i + x, j + y);
						Vec3b pix22 = X_mirrors.at<Vec3b>(i + x + 1, j + y + 1);
						G2 += abs(double(pix21[col]) - double(pix22[col]));
					}
				}

				Vec3b pix00 = X_mirrors.at<Vec3b>(i - 1, j - 1);
				Vec3b pix11 = X_mirrors.at<Vec3b>(i,     j    );
				Vec3b pix22 = X_mirrors.at<Vec3b>(i + 1, j + 1);
				Vec3b pix33 = X_mirrors.at<Vec3b>(i + 2, j + 2);
				Vec3b pix30 = X_mirrors.at<Vec3b>(i + 2, j - 1);
				Vec3b pix21 = X_mirrors.at<Vec3b>(i + 1, j    );
				Vec3b pix12 = X_mirrors.at<Vec3b>(i,     j + 1);
				Vec3b pix03 = X_mirrors.at<Vec3b>(i - 1, j + 2);

				if (100 * (1 + G1) > 100 * threshold*(1 + G2))
				{
					double val = attract(double(pix00[col]), double(pix11[col]), double(pix22[col]), double(pix33[col]), 1.);
					if (val < 0)
						val = 0;
					if (val > 255)
						val = 255;
					value[col] = val;
				}
				else if (100 * (1 + G2) > 100 * threshold*(1 + G1))
				{			
					double val = attract(double(pix30[col]), double(pix21[col]), double(pix12[col]), double(pix03[col]), 1.);
					if (val < 0)
						val = 0;
					if (val > 255)
						val = 255;
					value[col] = val;
				}
				else
				{
					double w1 = 1 / (1 + pow(G1, k));
					double w2 = 1 / (1 + pow(G2, k));
					double weight1 = w1 / (w1 + w2);
					double weight2 = w2 / (w1 + w2);
					double p1 = attract(double(pix00[col]), double(pix11[col]), double(pix22[col]), double(pix33[col]), 1. / 6.);
					double p2 = attract(double(pix30[col]), double(pix21[col]), double(pix12[col]), double(pix03[col]), 1. / 6.);
					double val = weight1 * p1 + weight2 * p2;
					if (val < 0)
						val = 0;
					if (val > 255)
						val = 255;
					value[col] = val;
				}
			}

			Result.at<Vec3b>(ii, jj) = value;
		}
	}

	copyMakeBorder(Result, R_mirrors, 4, 4, 4, 4, BORDER_REFLECT);

	for (int i = ws - 1; i < X_mirrors.rows - ws + 1; i++)
	{
		for (int j = ws - 1; j < X_mirrors.cols - ws + 1; j++)
		{
			//grey circle
			int ii1 = 2 * (i - ws + 1) + 1;
			int jj1 = 2 * (j - ws + 1) + 0;
			//white circle
			int ii2 = 2 * (i - ws + 1) + 0;
			int jj2 = 2 * (j - ws + 1) + 1;

			Vec3b value1;
			Vec3b value2;

			//fill grey circle
			for (int col = 0; col < 3; col++)
			{
				int x = ii1 + 4;
				int y = jj1 + 4;

				Vec3b square_left1   = R_mirrors.at<Vec3b>(x - 1, y - 2);
				Vec3b square_left2   = R_mirrors.at<Vec3b>(x + 1, y - 2);
				Vec3b circle_left1   = R_mirrors.at<Vec3b>(x - 2, y - 1);
				Vec3b circle_left2   = R_mirrors.at<Vec3b>(x    , y - 1);
				Vec3b circle_left3   = R_mirrors.at<Vec3b>(x + 2, y - 1);
				Vec3b square_middle1 = R_mirrors.at<Vec3b>(x - 1, y    );
				Vec3b square_middle2 = R_mirrors.at<Vec3b>(x + 1, y    );
				Vec3b circle_right1  = R_mirrors.at<Vec3b>(x - 2, y + 1);
				Vec3b circle_right2  = R_mirrors.at<Vec3b>(x    , y + 1);
				Vec3b circle_right3  = R_mirrors.at<Vec3b>(x + 2, y + 1);
				Vec3b square_right1  = R_mirrors.at<Vec3b>(x - 1, y + 2);
				Vec3b square_right2  = R_mirrors.at<Vec3b>(x + 1, y + 2);
				Vec3b more1          = R_mirrors.at<Vec3b>(x - 3, y    );
				Vec3b more2          = R_mirrors.at<Vec3b>(x + 3, y    );
				Vec3b more3          = R_mirrors.at<Vec3b>(x    , y + 3);
				Vec3b more4          = R_mirrors.at<Vec3b>(x    , y - 3);

				double G1 = abs(double(square_middle1[col]) - double(square_left1[col]) )  +
					        abs(double(square_right1[col])  - double(square_middle1[col])) +
					        abs(double(square_middle2[col]) - double(square_left2[col]))   +
					        abs(double(square_right2[col])  - double(square_middle2[col])) +
					        abs(double(more3[col])          - double(circle_right2[col]))  +
					        abs(double(circle_left2[col])   - double(more4[col]))          +
					        abs(double(circle_right1[col])  - double(circle_left1[col]))   +
					        abs(double(circle_right2[col])  - double(circle_left2[col]))   +
					        abs(double(circle_right3[col])  - double(circle_left3[col]));

				double G2 = abs(double(square_left1[col])   - double(square_left2[col]))   +
					        abs(double(square_middle1[col]) - double(square_middle2[col])) +
					        abs(double(square_right2[col])  - double(square_right1[col]))  +
					        abs(double(more1[col])          - double(square_middle1[col])) +
					        abs(double(square_middle2[col]) - double(more2[col]) )         +
					        abs(double(circle_left1[col])   - double(circle_left2[col]))   +
					        abs(double(circle_left2[col])   - double(circle_left3[col]))   +
					        abs(double(circle_right1[col])  - double(circle_right2[col]))  +
					        abs(double(circle_right2[col])  - double(circle_right3[col]));

				if (100 * (1 + G1) > 100 * threshold*(1 + G2))
				{
					double val = attract(double(more1[col]), double(square_middle1[col]), double(square_middle2[col]), double(more2[col]), 1.);
					if (val < 0)
						val = 0;
					if (val > 255)
						val = 255;
					value1[col] = val;
				}

				else if (100 * (1 + G2) > 100 * threshold*(1 + G1))
				{
					double val = attract(double(more4[col]), double(circle_left2[col]), double(circle_right2[col]), double(more3[col]), 1.);
					if (val < 0)
						val = 0;
					if (val > 255)
						val = 255;
					value1[col] = val;
				}
				else
				{
					double w1 = 1 / (1 + pow(G1, k));
					double w2 = 1 / (1 + pow(G2, k));
					double weight1 = w1 / (w1 + w2);
					double weight2 = w2 / (w1 + w2);
					double p1 = attract(double(more1[col]), double(square_middle1[col]), double(square_middle2[col]), double(more2[col]), 1. / 6.);
					double p2 = attract(double(more4[col]), double(circle_left2[col]), double(circle_right2[col]), double(more3[col]), 1. / 6.);
					double val = weight1 * p1 + weight2 * p2;
					if (val < 0)
						val = 0;
					if (val > 255)
						val = 255;
					value1[col] = val;
				}
			}

			//fill white circle
			for (int col = 0; col < 3; col++)
			{
				int x = ii2 + 4;
				int y = jj2 + 4;

				Vec3b circle_left1   = R_mirrors.at<Vec3b>(x - 1, y - 2);
				Vec3b circle_left2   = R_mirrors.at<Vec3b>(x + 1, y - 2);
				Vec3b square_left1   = R_mirrors.at<Vec3b>(x - 2, y - 1);
				Vec3b square_left2   = R_mirrors.at<Vec3b>(x    , y - 1);
				Vec3b square_left3   = R_mirrors.at<Vec3b>(x + 2, y - 1);
				Vec3b circle_middle1 = R_mirrors.at<Vec3b>(x - 1, y - 0);
				Vec3b circle_middle2 = R_mirrors.at<Vec3b>(x + 1, y - 0);
				Vec3b square_right1  = R_mirrors.at<Vec3b>(x - 2, y + 1);
				Vec3b square_right2  = R_mirrors.at<Vec3b>(x    , y + 1);
				Vec3b square_right3  = R_mirrors.at<Vec3b>(x + 2, y + 1);
				Vec3b circle_right1  = R_mirrors.at<Vec3b>(x - 1, y + 2);
				Vec3b circle_right2  = R_mirrors.at<Vec3b>(x + 1, y + 2);
				Vec3b more1          = R_mirrors.at<Vec3b>(x - 3, y    );
				Vec3b more2          = R_mirrors.at<Vec3b>(x + 3, y    );
				Vec3b more3          = R_mirrors.at<Vec3b>(x    , y + 3);
				Vec3b more4          = R_mirrors.at<Vec3b>(x    , y - 3);

				double G1 = abs(double(circle_middle1[col]) - double(circle_left1[col]))   +
					        abs(double(circle_right1[col])  - double(circle_middle1[col])) +
					        abs(double(circle_middle2[col]) - double(circle_left2[col]))   +
					        abs(double(circle_right2[col])  - double(circle_middle2[col])) +
					        abs(double(more3[col])          - double(square_right2[col]))  +
					        abs(double(square_left2[col])   - double(more4[col]))          +
					        abs(double(square_right1[col])  - double(square_left1[col]) )  +
					        abs(double(square_right2[col])  - double(square_left2[col]))   +
					        abs(double(square_right3[col])  - double(square_left3[col]));

				double G2 = abs(double(circle_left1[col])   - double(circle_left2[col]))   +
					        abs(double(circle_middle1[col]) - double(circle_middle2[col])) +
					        abs(double(circle_right2[col])  - double(circle_right1[col]))  +
					        abs(double(more1[col])          - double(circle_middle1[col])) +
					        abs(double(circle_middle2[col]) - double(more2[col]))          +
					        abs(double(square_left1[col])   - double(square_left2[col]))   +
					        abs(double(square_left2[col])   - double(square_left3[col]))   +
					        abs(double(square_right1[col])  - double(square_right2[col]))  +
					        abs(double(square_right2[col])  - double(square_right3[col]));

				if (100 * (1 + G1) > 100 * threshold*(1 + G2))
				{
					double val = attract(double(more1[col]), double(circle_middle1[col]), double(circle_middle2[col]), double(more2[col]), 1.);
					if (val < 0)
						val = 0;
					if (val > 255)
						val = 255;
					value2[col] = val;
				}

				else if (100 * (1 + G2) > 100 * threshold*(1 + G1))
				{
					double val = attract(double(more4[col]), double(square_left2[col]), double(square_right2[col]), double(more3[col]), 1.);
					if (val < 0)
						val = 0;
					if (val > 255)
						val = 255;
					value2[col] = val;
				}
				else
				{
					double w1 = 1 / (1 + pow(G1, k));
					double w2 = 1 / (1 + pow(G2, k));
					double weight1 = w1 / (w1 + w2);
					double weight2 = w2 / (w1 + w2);
					double p1 = attract(double(more1[col]), double(circle_middle1[col]), double(circle_middle2[col]), double(more2[col]), 1. / 6.);
					double p2 = attract(double(more4[col]), double(square_left2[col]), double(square_right2[col]), double(more3[col]), 1. / 6.);
					double val = weight1 * p1 + weight2 * p2;
					if (val < 0)
						val = 0;
					if (val > 255)
						val = 255;
					value2[col] = val;
				}
			}

			Result.at<Vec3b>(ii1, jj1) = value1;
			Result.at<Vec3b>(ii2, jj2) = value2;
		}
	}

	cout << "test2" << endl;
	return Result;
}


int main()
{
	Mat src;

	char file[40] = "D:/engle/image_zoom/image_zoom/lena.jpg";
	src = imread(file);
	if (!src.data)
		return -1;
	
	Mat X = interpolate_rgb(src);
	char dst[41] = "D:/engle/image_zoom/image_zoom/lena2.jpg";
	imwrite(dst, X);
	waitKey(0);
}
