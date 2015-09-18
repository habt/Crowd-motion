/*
 * Optical_flow_matlab.cpp
 *
 *  Created on: Aug 19, 2013
 *      Author: habte
 */
#include <iostream>
#include <iomanip>
#include <string>

#include <stdio.h>
#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>
#include "opencv2/opencv.hpp"
#include"Functions.h"
#include <math.h>
#include "opencv2/gpu/gpu.hpp"
#include<cmath>
#include "engine.h"
#include <string.h>


//#include "cvconfig.h"
#include "opencv2/core/core.hpp"
#include "opencv2/core/opengl_interop.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

static const double pi = 3.14159265358979323846;

using namespace cv;
using namespace std;
using namespace cv::gpu;


int main()
{
	Engine *ep = engOpen(NULL);
		CvCapture *input_video = cvCaptureFromFile("MeccaSequence_WithSyntheticInstability.mpg");
		double _height = (int) cvGetCaptureProperty(input_video, CV_CAP_PROP_FRAME_HEIGHT);
		double _width = (int) cvGetCaptureProperty(input_video, CV_CAP_PROP_FRAME_WIDTH);
		double fr_count = (int) cvGetCaptureProperty(input_video, CV_CAP_PROP_FRAME_COUNT);

		int framenumber = (int) cvGetCaptureProperty(input_video, CV_CAP_PROP_FRAME_COUNT);

		int fr_height = (int)_height;
		int fr_width =(int)_width;

		double rate = cvGetCaptureProperty(input_video , CV_CAP_PROP_FPS);
		cout<<endl<<"frame rate is "<< rate<<endl;
		//cvWaitKey(0);


		vector<Mat> fx(framenumber);
		vector<Mat> fy(framenumber);
		vector<Mat> angles(framenumber);

		int scale =3;
		int slct_width = int(fr_width/scale);
		int slct_height = int(fr_height/scale);

		double x_opt_init[fr_height][fr_width];
		double y_opt_init[fr_height][fr_width];

		mxArray *xopt_flw_array = mxCreateDoubleMatrix(fr_height,fr_width,mxREAL);//rows,columns
		mxArray *yopt_flw_array = mxCreateDoubleMatrix(fr_height,fr_width,mxREAL);

		//double *xpopt = mxGetPr(xopt_flw_array);
		//double *ypopt = mxGetPr(yopt_flw_array);

		printf("\n height * width of selected is %d*%d \n",fr_height,fr_width);
		/*go to frame N*/
		int frame_count =0;

		IplImage *frame = cvQueryFrame(input_video);
		//if (frame == NULL) break;


		while (frame_count<27) // or number of frames
		{
			CvSize cvsize_frame,cvsize_wind;

			cvsize_frame.height = fr_height;
			cvsize_frame.width = fr_width;

			cvsize_wind.height = 5;
			cvsize_wind.width = 5;


			IplImage *firstframe = cvCreateImage(cvSize(fr_width,fr_height),IPL_DEPTH_8U, 1);
			cvConvertImage(frame, firstframe,0);

			cvShowImage("frame 1 gray",firstframe);

		    //get next frame
			frame = cvQueryFrame( input_video );
			if (frame == NULL) break;
								//allocateOnDemand( &frame2_1C, frame_size, IPL_DEPTH_8U, 1 );
			IplImage *secondframe = cvCreateImage(cvSize(fr_width,fr_height),IPL_DEPTH_8U, 1);
			cvConvertImage(frame, secondframe, 0);

			cvShowImage("frame 2 gray",secondframe);


            // project worked with this optical flow
			//Calculate optical flow using the sparse lucas-kanade algorithm - less accurate but fast
			IplImage* velX= cvCreateImage(cvsize_frame,IPL_DEPTH_32F,1);
			IplImage* velY= cvCreateImage(cvsize_frame,IPL_DEPTH_32F,1);
			cvCalcOpticalFlowLK(firstframe,secondframe,cvsize_wind,velX,velY);

			Mat velx32(velX);
			Mat vely32(velY);

			Mat velxmat,velymat;

			velx32.clone().convertTo(velxmat,CV_64FC1);
			vely32.clone().convertTo(velymat,CV_64FC1);



/*
			//calculate optical flow using the dense Gunnar Farneback’s algorithm -
			//more accurate but slow and not producing results at the moment

			Mat velmat2c(cvsize_frame, CV_32FC2);
			Mat previous(firstframe);
			Mat next(firstframe);


			calcOpticalFlowFarneback(previous,next,velmat2c,0.5,1,10,20,5,1.0,1 ); // parameters herecan change

			//split the two channel velocity mat in to x channel and y channel

			Mat velxmat,velymat;
			Mat velarr[2];
			split(velmat2c,velarr); //index 0 of velarr is x velocity matrix, index 1 is y velocity matrix.

			//convert floating point output to double
			velarr[0].clone().convertTo(velxmat,CV_64FC1);
			velarr[1].clone().convertTo(velymat,CV_64FC1);

           //write the optical flow in to an xml file to check with the one sent to matlab
			if (frame_count ==26)
			{
				cout<<endl<<velxmat<<endl;
				FileStorage f;
				f.open("temp.xml", FileStorage::WRITE);
				f << "velxmat" << velxmat;
				f.release();
			}
*/

/*
	   //brox optical flow part

			// Load images
    	Mat PreviousFrame(firstframe); // Has an image in format CV_32FC1
    	Mat CurrentFrame(secondframe);  // Has an image in format CV_32FC1

    	Mat PreviousFrameGrayFloat; // Has an image in format CV_32FC1
    	Mat CurrentFrameGrayFloat;  // Has an image in format CV_32FC1

    	PreviousFrame.convertTo(PreviousFrameGrayFloat,CV_32F,1.0 / 255.0);
    	CurrentFrame.convertTo(CurrentFrameGrayFloat,CV_32F, 1.0 / 255.0);



    	// Upload images to GPU
    	const cv::gpu::GpuMat PreviousFrameGPU(PreviousFrameGrayFloat);
    	const cv::gpu::GpuMat CurrentFrameGPU(CurrentFrameGrayFloat);

    	// Prepare receiving variables
    	cv::gpu::GpuMat FlowXGPU = GpuMat(fr_height, fr_width, CV_32F);
    	cv::gpu::GpuMat FlowYGPU = GpuMat(fr_height, fr_width, CV_32F);

    	// Create optical flow object
    	cv::gpu::BroxOpticalFlow OpticalFlowGPU = BroxOpticalFlow(0.197f,50.0f,0.8f, 10, 77, 10);

    	// Perform optical flow
    	OpticalFlowGPU(PreviousFrameGPU, CurrentFrameGPU, FlowXGPU, FlowYGPU);
    	// Exception in opencv_core244d!cv::GlBuffer::unbind

    	// Download flow from GPU
    	Mat velxmat;
    	Mat velymat;
    	FlowXGPU.download(velxmat);
    	FlowYGPU.download(velymat);
*/



			for (int i=0;i<fr_height;i++)
			{
			    for(int j=0;j<fr_width;j++)
			    {
			    	x_opt_init[i][j]=velxmat.at<double>(i,j);
			      //xpopt[i][j]=xopt[i][j];
			    	y_opt_init[i][j]=velymat.at<double>(i,j);
			      //ypopt[i][j]=yopt[i][j];
			    }
			}

			if (frame_count ==26) cout<<endl<<x_opt_init[fr_height-1][fr_width-1]<<endl;

			memcpy(mxGetPr(xopt_flw_array), x_opt_init, sizeof(x_opt_init));
			memcpy(mxGetPr(yopt_flw_array), y_opt_init, sizeof(y_opt_init));

			//memcpy(mxGetData(xpopt), xopt, sizeof(xopt));


			if(frame_count==0)
			{
				engPutVariable(ep,"uarr",xopt_flw_array);
				engPutVariable(ep,"varr",yopt_flw_array);
				engEvalString(ep,"uarr = reshape(uarr,size(uarr,2),size(uarr,1))");
				engEvalString(ep,"varr = reshape(varr,size(varr,2),size(varr,1))");
				engEvalString(ep,"uarr = transpose(uarr)");
				engEvalString(ep,"varr = transpose(varr)");


			}
			else
			{
				engPutVariable(ep,"uin",xopt_flw_array);
				engPutVariable(ep,"vin",yopt_flw_array);
				engEvalString(ep,"uin = reshape(uin,size(uin,2),size(uin,1))");
				engEvalString(ep,"vin = reshape(vin,size(vin,2),size(vin,1))");
				engEvalString(ep,"uin = transpose(uin)");
				engEvalString(ep,"vin = transpose(vin)");
				engEvalString(ep,"uarr = cat(3,uarr,uin)");//concatenate  C = cat(dim, A, B)
				engEvalString(ep,"varr = cat(3,varr,vin)");
			}

			//mxDestroyArray();
			frame_count++;
			cout << endl<<"inside while loop with frame count " <<frame_count<<"  from " <<framenumber<< endl;
			if(frame_count==25)
			{
				double *c2result;//[fr_height][fr_width];
				mxArray *m2result = engGetVariable(ep,"uin");
				c2result = mxGetPr(m2result);
				cout<<"returned uarr value is is"<<endl<<c2result[3353]<<endl;
			}
			cvConvertImage(secondframe,frame, 0);

		}



		Mat xinit = initXO(fr_height,fr_width);
		Mat yinit = initYO(fr_height,fr_width);


		//double* px = mxGetPr(xinit_array);
		//double* py = mxGetPr(yinit_array);
		//double *pfr = mxGetPr(frame_rate);
		mxArray *xinit_array = mxCreateDoubleMatrix(fr_height,fr_width,mxREAL);
		mxArray *yinit_array = mxCreateDoubleMatrix(fr_height,fr_width,mxREAL);
		mxArray *frame_rate = mxCreateDoubleMatrix(1,1,mxREAL);



		for (int i=0;i<fr_height;i++)
			{
				for(int j=0;j<fr_width;j++)
					{
					x_opt_init[i][j]=xinit.at<double>(i,j);
					     //px[i][j]=xopt[i][j];
					y_opt_init[i][j]=yinit.at<double>(i,j);
					     //py[i][j]=yopt[i][j];
					  }
			 }

		memcpy(mxGetPr(xinit_array), x_opt_init, sizeof(double)*fr_height*fr_width);
		memcpy(mxGetPr(yinit_array), y_opt_init, sizeof(double)*fr_height*fr_width);
		memcpy(mxGetPr(frame_rate), &rate, sizeof(double));

		engPutVariable(ep,"xin",xinit_array);
		engPutVariable(ep,"yin",xinit_array);
		engPutVariable(ep,"fps",frame_rate);

		double *cresult;//[fr_height][fr_width];
		mxArray *mresult = engGetVariable(ep,"fps");
		cresult = mxGetPr(mresult);
		cout<<"returned fps is"<<endl<<cresult[0]<<endl; //480*640  --when we ask for cresult[700] it gives 700
		//cvWaitKey(0);                                        // //cresult[1000] gives 360
														//cresult[307199] gives 639 - outside scope after this number

		cout << endl<<"before last eval string call " << endl;
		//cvWaitKey(0);

		engEvalString(ep,"go_segmentation");


		/*double *c2result;//[fr_height][fr_width];
		engEvalString(ep,"uo=uarr(473,633,27)");
		mxArray *m2result = engGetVariable(ep,"uo");
		c2result = mxGetPr(m2result);
		cout<<"returned end_frame is"<<endl<<c2result[0]<<endl;*/

		return 0;

		// in matlab we have variables uarr and varr as x and y optivcal flow
		//we have xin and yin as x and y initial
		//we have rate as frame rate of video

}



