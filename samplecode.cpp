#include "arcsoft_face_sdk.h"
#include "amcomdef.h"
#include "asvloffscreen.h"
#include "merror.h"
#include <iostream>  
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define APPID "HV7iMti7TD73cKwPvqVnVoj8FPctY8y9wJNaQxmFUCTG"
#define SDKKEY "GrZv23NEm1LeNr5W5QpX9dZzSeavGwMhA4W4B92DeYaw"

#define SafeFree(p) { if ((p)) free(p); (p) = NULL; }
#define SafeArrayDelete(p) { if ((p)) delete [] (p); (p) = NULL; } 
#define SafeDelete(p) { if ((p)) delete (p); (p) = NULL; } 

// #define NSCALE "取值范围[2,32]，VIDEO模式推荐值16，IMAGE模式内部设置固定值为30" 
// #define FACENUM	"检测的人脸数"
#define NSCALE 16
#define FACENUM 100
	
MHandle handle = NULL;
MRESULT res;

void mark_faces(cv::Mat &img,ASF_MultiFaceInfo &facesPosData,std::vector<cv::Point3f> facesAngle){
	cv::Rect _position;
    for(int i = 0;i<facesPosData.faceNum;i++){
        string text = cv::format("(%0.2f,%0.2f,%0.2f)",facesAngle[i].x,facesAngle[i].y,facesAngle[i].z);
        putText(img,text,Point(facesPosData.faceRect[i].left,facesPosData.faceRect[i].top - 5),FONT_HERSHEY_PLAIN,1,Scalar(0,0,255));//显示解算角度

		_position.x = facesPosData.faceRect[i].left;
		_position.y = facesPosData.faceRect[i].top;
		_position.height = facesPosData.faceRect[i].bottom - facesPosData.faceRect[i].top;
		_position.width = facesPosData.faceRect[i].right - facesPosData.faceRect[i].left;

        rectangle(img,_position,Scalar(255,0,0),2,8);
    }
}

cv::Point3f get_Ang(int center_x,int center_y,int width){
	double calibration[9] = {
		945.2654, 0.000000, 701.7872,
		0.000000, 945.6360, 319.5665,
		0.000000, 0.000000, 1.000000
	};
	double dist_coeffs[5] = { -0.0112, 0.0878, 0.0000, 0.0000, -0.0992 };
	Mat cameraMatrix = Mat(3, 3, CV_64F, calibration);	//内参矩阵
	Mat distCoeffs = Mat(5, 1, CV_64F, dist_coeffs);	//畸变系数
	
	vector<Point3f> object_point;
	object_point.push_back(Point3f(-80, -80, 0.0));	//人脸宽80，以中心为原点
	object_point.push_back(Point3f(80,  -80, 0.0));
	object_point.push_back(Point3f(80,  80 , 0.0));
	object_point.push_back(Point3f(-80, 80 , 0.0));

	vector<Point2f> image_point;
	image_point.push_back(Point2f(center_x - width, center_y - width));
	image_point.push_back(Point2f(center_x + width, center_y - width));
	image_point.push_back(Point2f(center_x + width, center_y + width));
	image_point.push_back(Point2f(center_x - width, center_y + width));

	Mat rvec = Mat::ones(3, 1, CV_64F);	//旋转矩阵
	Mat tvec = Mat::ones(3, 1, CV_64F);	//平移矩阵

	solvePnP(object_point, image_point, cameraMatrix, distCoeffs, rvec, tvec);
	
	double pos_x, pos_y, pos_z;
	const double *_xyz = (const double *)tvec.data;
	pos_z = tvec.at<double>(2) / 1000.0;
	pos_x = atan2(_xyz[0], _xyz[2]);
	pos_y = atan2(_xyz[1], _xyz[2]);
	pos_x *= 180 / 3.1415926;
	pos_y *= 180 / 3.1415926;

	Point3f pos = Point3f(pos_x,pos_y,pos_z);
	return pos;
}

void init_Engine(){
/* 	//激活SDK
	res = ASFOnlineActivation(APPID, SDKKEY);
	if (MOK != res && MERR_ASF_ALREADY_ACTIVATED != res)
		printf("ASFOnlineActivation fail: %d\n", res);
	else
		printf("ASFOnlineActivation sucess: %d\n", res); */

	//初始化引擎
	handle = NULL;	//引擎句柄
	MInt32 mask = ASF_FACE_DETECT | ASF_FACERECOGNITION | ASF_AGE | ASF_GENDER | ASF_FACE3DANGLE | ASF_LIVENESS | ASF_IR_LIVENESS;	//需要启用的功能组合
	res = ASFInitEngine(ASF_DETECT_MODE_IMAGE, ASF_OP_0_ONLY, NSCALE, FACENUM, mask, &handle);	//初始化引擎
	if (res != MOK)
		printf("ALInitEngine fail: %d\n", res);
	else
		printf("ALInitEngine sucess: %d\n", res);
}

int main()
{
	init_Engine();
	cv::VideoCapture cap(0);
	cv::Mat frame;
	while(1){
		cap >> frame;
		cv::imwrite("temp.jpg",frame);

		// IplImage* img1 = cvLoadImage("temp.jpg");//图片宽度需符合4的倍数
		// IplImage* img2 = cvLoadImage("2.bmp");

		// if (img1 && img2){
			ASF_MultiFaceInfo detectedFaces = { 0 };//多人脸信息；
			// ASF_SingleFaceInfo SingleDetectedFaces1 = { 0 };
			// ASF_FaceFeature feature1 = { 0 };
			// ASF_FaceFeature copyfeature1 = { 0 };

			// res = ASFDetectFaces(handle, img1->width, img1->height, ASVL_PAF_RGB24_B8G8R8, (MUInt8*)img1->imageData, &detectedFaces1);
			res = ASFDetectFaces(handle, frame.cols, frame.rows, ASVL_PAF_RGB24_B8G8R8, (MUInt8*)frame.data, &detectedFaces);

			if (MOK == res){
				std::vector<cv::Point3f> faceAngle; //人脸在世界坐标的位置
				float minDis = 1024;                //在摄像头前的人脸最小距离
				for(int i = 0;i<detectedFaces.faceNum;i++){
					//计算出每个人脸中心坐标，并进行PNP解算,找出每张脸在世界坐标的位置
					int center_x,center_y,width;
					center_x = (detectedFaces.faceRect[i].left + detectedFaces.faceRect[i].right) / 2;
					center_y = (detectedFaces.faceRect[i].bottom + detectedFaces.faceRect[i].top) / 2;
					width = detectedFaces.faceRect[i].right - detectedFaces.faceRect[i].left;

					faceAngle.push_back(get_Ang(center_x,center_y,width));
					if(faceAngle[i].z < minDis)
						minDis = faceAngle[i].z;
				}
				cout<<"MiniDis:"<<minDis<<endl;

				
				mark_faces(frame,detectedFaces,faceAngle);

				//人脸信息检测
/* 				MInt32 processMask = ASF_AGE | ASF_GENDER | ASF_FACE3DANGLE;
				res = ASFProcess(handle, frame.cols, frame.rows, ASVL_PAF_RGB24_B8G8R8, (MUInt8*)frame.data, &detectedFaces, processMask);
				if (res != MOK)
					cout<<"img2人脸信息检测失败，返回码: " << (int)res;
				else
					cout<<"img2人脸信息检测成功，返回码: " << (int)res; 
				//获取3D信息
				

 				SingleDetectedFaces1.faceRect.left = detectedFaces1.faceRect[0].left;
				SingleDetectedFaces1.faceRect.top = detectedFaces1.faceRect[0].top;
				SingleDetectedFaces1.faceRect.right = detectedFaces1.faceRect[0].right;
				SingleDetectedFaces1.faceRect.bottom = detectedFaces1.faceRect[0].bottom;
				SingleDetectedFaces1.faceOrient = detectedFaces1.faceOrient[0];

				//单人脸特征提取
				res = ASFFaceFeatureExtract(handle, img1->width, img1->height, ASVL_PAF_RGB24_B8G8R8, (MUInt8*)img1->imageData, &SingleDetectedFaces1, &feature1);
				if (res == MOK){
					//拷贝feature
					copyfeature1.featureSize = feature1.featureSize;
					copyfeature1.feature = (MByte *)malloc(feature1.featureSize);
					memset(copyfeature1.feature, 0, feature1.featureSize);
					memcpy(copyfeature1.feature, feature1.feature, feature1.featureSize);
					cout<<"img1面部特征提取成功，返回码: "<<(int)res<<endl;
				}
				else 
					cout<<"img1面部特征提取失败，返回码: "<<(int)res<<endl;*/
			}
			else
				cout<<"img1面部特征提取失败，返回码"<<(int)res<<endl;
			cv::imshow("detect",frame);
			char c = cv::waitKey(10);
			if(c==27)
				return 0;
		// }
	}

/* 	//第二张人脸提取特征
	ASF_MultiFaceInfo	detectedFaces2 = { 0 };
	ASF_SingleFaceInfo SingleDetectedFaces2 = { 0 };
	ASF_FaceFeature feature2 = { 0 };
	res = ASFDetectFaces(handle, img2->width, img2->height, ASVL_PAF_RGB24_B8G8R8, (MUInt8*)img2->imageData, &detectedFaces2);
	if (MOK == res)
	{
		SingleDetectedFaces2.faceRect.left = detectedFaces2.faceRect[0].left;
		SingleDetectedFaces2.faceRect.top = detectedFaces2.faceRect[0].top;
		SingleDetectedFaces2.faceRect.right = detectedFaces2.faceRect[0].right;
		SingleDetectedFaces2.faceRect.bottom = detectedFaces2.faceRect[0].bottom;
		SingleDetectedFaces2.faceOrient = detectedFaces2.faceOrient[0];

		res = ASFFaceFeatureExtract(handle, img2->width, img2->height, ASVL_PAF_RGB24_B8G8R8, (MUInt8*)img2->imageData, &SingleDetectedFaces2, &feature2);
		if (MOK != res)
		ui->textEdit_State->append("img2面部特征提取失败，返回码: " + QString::number(res) );
		else
		ui->textEdit_State->append("img2面部特征提取成功，返回码: " + QString::number(res) );
	}
	else
	ui->textEdit_State->append("img2面部特征提取失败，返回码: " + QString::number(res) );
 */
}
