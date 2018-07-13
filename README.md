# image-magnifying
This repo implements an image magnifying algorithm: Directional Cubic Convolution Interpolation.

It contains both CPU and GPU(CUDA) versions.

For the algorithm, I suggest reading: Image zooming using directional cubic convolution interpolation by Dr. Zhou.
In total, it contains three passes to fill the holes.

Original picture:

![alt text](lena.jpg)

2X magnify CPU version:

![alt text](lena2.jpg)

2X magnify GPU version:

![alt text](lena3.jpg)
