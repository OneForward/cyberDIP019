#include "usrGameController.h"

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
using namespace std;
using namespace cv;

Point* resultPoints;

void matchTemplate(int);
void matchTemplate(int, int);
void clipOriginPic(int mode);

#ifdef VIA_OPENCV
//构造与初始化
usrGameController::usrGameController(void* qtCD)
{
	qDebug() << "usrGameController online.";
	device = new deviceCyberDip(qtCD);//设备代理类
	cv::namedWindow(WIN_NAME);
	cv::setMouseCallback(WIN_NAME, mouseCallback, (void*)&(argM));
	counter = 0;
}

//析构
usrGameController::~usrGameController()
{
	cv::destroyAllWindows();
	if (device != nullptr)
	{
		delete device;
	}
	qDebug() << "usrGameController offline.";
}

//处理图像 
int usrGameController::usrProcessImage(cv::Mat& img)
{
	cv::Size imgSize(img.cols, img.rows - UP_CUT);
	if (imgSize.height <= 0 || imgSize.width <= 0)
	{
		qDebug() << "Invalid image. Size:" << imgSize.width <<"x"<<imgSize.height;
		return -1;
	}

	//截取图像边缘
	cv::Mat pt = img(cv::Rect(0, UP_CUT, imgSize.width,imgSize.height));
	cv::imshow(WIN_NAME, pt);
	
	//判断鼠标点击尺寸
	if (argM.box.x >= 0 && argM.box.x < imgSize.width&&
		argM.box.y >= 0 && argM.box.y < imgSize.height
		)
	{
		qDebug() << "X:" << argM.box.x << " Y:" << argM.box.y;
		if (argM.Hit)
		{
			device->comHitDown();
		}
		device->comMoveToScale(((double)argM.box.x + argM.box.width) / pt.cols, ((double)argM.box.y + argM.box.height) / pt.rows);
		argM.box.x = -1; argM.box.y = -1;
		if (argM.Hit)
		{
			device->comHitUp();
		}
		else
		{
			device->comHitOnce();
		}
	}

	/*************以下为我添加的代码******************/
	counter++;
	int mode = 8;// 8*8=64模式
	string fname = "E:/frames/frame_"; 
	fname += ('0' + mode*mode / 10); fname += ('0' + mode*mode % 10); fname += '_';
	fname += ('0' + counter / 100); fname += ('0' + counter / 10 % 10); fname += ('0' + counter % 10);
	fname += (".png");
	imwrite(fname, img);
	//clipOriginPic(mode);
	matchTemplate(mode);
	//matchTemplate(mode, 3);







	return 0; 
}

//鼠标回调函数
void mouseCallback(int event, int x, int y, int flags, void*param)
{
	usrGameController::MouseArgs* m_arg = (usrGameController::MouseArgs*)param;
	switch (event)
	{
	case CV_EVENT_MOUSEMOVE: // 鼠标移动时
	{
		if (m_arg->Drawing)
		{
			m_arg->box.width = x - m_arg->box.x;
			m_arg->box.height = y - m_arg->box.y;
		}
	}
	break;
	case CV_EVENT_LBUTTONDOWN:case CV_EVENT_RBUTTONDOWN: // 左/右键按下
	{
		m_arg->Hit = event == CV_EVENT_RBUTTONDOWN;
		m_arg->Drawing = true;
		m_arg->box = cvRect(x, y, 0, 0);
	}
	break;
	case CV_EVENT_LBUTTONUP:case CV_EVENT_RBUTTONUP: // 左/右键弹起
	{
		m_arg->Hit = false;
		m_arg->Drawing = false;
		if (m_arg->box.width < 0)
		{
			m_arg->box.x += m_arg->box.width;
			m_arg->box.width *= -1;
		}
		if (m_arg->box.height < 0)
		{
			m_arg->box.y += m_arg->box.height;
			m_arg->box.height *= -1;
		}
	}
	break;
	}
}

/*******segmentation***************/
void clipOriginPic(int mode) {
	//const int x0 = 92, y0 = 512, x1 = 992, y1 = 1412;// 针对我手机截图的标定
	const int x0 = 52, y0 = 315, x1 = 489, y1 = 752;// 针对screencapture
	int mode_square = mode*mode;
	int *xi0 = new int[mode_square];
	int *yi0 = new int[mode_square];
	int *xi1 = new int[mode_square];
	int *yi1 = new int[mode_square];
	string file_init = "E:/play00_init.png";
	file_init[7] = mode_square / 10 + '0';
	file_init[8] = mode_square % 10 + '0';
	Mat src = imread(file_init);

	rectangle(src, Point(x0, y0), Point(x1, y1), Scalar(0, 0, 255));
	int d = (x1 - x0) / 4 / mode;
	namedWindow("Play InitImage");
	imshow("Play InitImage", src);
	Mat *seg = new Mat[mode_square];
	Mat *seg_output = new Mat[mode_square];
	char file_seg[] = "E:/seg00_small_00.png";
	file_seg[6] = mode_square / 10 + '0';
	file_seg[7] = mode_square % 10 + '0';
	for (int i = 0; i < mode_square; ++i) {
		xi0[i] = ((i % mode) * 4 + 1) *d + x0;
		yi0[i] = ((i / mode) * 4 + 1) * d + y0;
		seg[i] = src(Rect(xi0[i], yi0[i], 2 * d, 2 * d));
		file_seg[15] = '0' + i / 10;
		file_seg[16] = '0' + i % 10;
		seg[i].copyTo(seg_output[i]);
		imwrite(file_seg, seg_output[i]);
		cout << "成功写入img" << file_seg << endl;
	}
}




/****************模板匹配***************/
void matchTemplate(int mode) {
	Mat srcImage, templImage, img, templ, result;
	int match_method;
	string file_play("E:/play00.png"), file_seg("E:/seg00_small_00.png"), file_final("E:/final00.png");
	file_play[7] = mode*mode / 10 + '0';
	file_play[8] = mode*mode % 10 + '0';
	file_seg[6] = mode*mode / 10 + '0';
	file_seg[7] = mode*mode % 10 + '0';
	file_final[8] = mode*mode / 10 + '0';
	file_final[9] = mode*mode % 10 + '0';

	srcImage = imread(file_play, IMREAD_COLOR);

	float alpha = 2.5;
	resize(srcImage, img, Size(srcImage.cols / alpha, srcImage.rows / alpha));

	namedWindow("Source Image");

	Mat img_display;
	img.copyTo(img_display);
	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;
	result.create(result_rows, result_cols, CV_32FC1);
	match_method = TM_SQDIFF;

	resultPoints = new Point[mode*mode];
	string num = "00";
	for (int i = 0; i < mode*mode; ++i) {
		num[0] = '0' + i / 10;
		num[1] = '0' + i % 10;
		templ.release(); result.release();
		file_seg[15] = num[0];
		file_seg[16] = num[1];
		templImage = imread(file_seg, IMREAD_COLOR);
		resize(templImage, templ, Size(templImage.cols / alpha, templImage.rows / alpha));
		matchTemplate(img, templ, result, match_method);
		normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
		double minVal; double maxVal; Point minLoc; Point maxLoc;
		Point matchLoc;
		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
		matchLoc = minLoc;
		cout << "第" << i << "个拼图的参数\nminValue=" << minVal << "\tminLoc=(" << minLoc.x << ", " << minLoc.y << ")\n";

		resultPoints[i] = Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows);
		rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(255), 2, 8, 0);
		putText(img_display, num, Point(matchLoc.x, matchLoc.y + templ.rows / 1.5), 6, 1, Scalar(255, 0, 255), 2);
	}

	imshow("Source Image", img);
	imwrite(file_final, img_display);
	namedWindow(file_final);
	imshow(file_final, img_display);

	//namedWindow("template", WINDOW_AUTOSIZE);
	//imshow("template", templ);

	//本函数可提供一个全局接口 resultPoints[mode*mode] 即最终所有拼图子块的中心
	//结果不完全正确，可以先取相似度最高的子图先移动，再逐步调整
}

//模板匹配接口2
void matchTemplate(int mode, int i) {
	Mat srcImage, templImage, img, templ, result;
	int match_method;
	string file_play("E:/play00.png"), file_seg("E:/seg00_small_00.png"), file_final("E:/final00.png");
	file_play[7] = mode*mode / 10 + '0';
	file_play[8] = mode*mode % 10 + '0';
	file_seg[6] = mode*mode / 10 + '0';
	file_seg[7] = mode*mode % 10 + '0';
	file_final[8] = mode*mode / 10 + '0';
	file_final[9] = mode*mode % 10 + '0';

	srcImage = imread(file_play, IMREAD_COLOR);

	float alpha = 2.5;
	resize(srcImage, img, Size(srcImage.cols / alpha, srcImage.rows / alpha));

	namedWindow("Source Image");

	Mat img_display;
	img.copyTo(img_display);
	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;
	result.create(result_rows, result_cols, CV_32FC1);
	match_method = TM_SQDIFF;

	//resultPoints = new Point[mode*mode];
	string num = "00";

	num[0] = '0' + i / 10;
	num[1] = '0' + i % 10;
	templ.release(); result.release();
	file_seg[15] = num[0];
	file_seg[16] = num[1];
	templImage = imread(file_seg, IMREAD_COLOR);
	resize(templImage, templ, Size(templImage.cols / alpha, templImage.rows / alpha));
	matchTemplate(img, templ, result, match_method);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	matchLoc = minLoc;
	cout << "第" << i << "个拼图的参数\nminValue=" << minVal << "\tminLoc=(" << minLoc.x << ", " << minLoc.y << ")\n";

	/*resultPoints[i] = Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows);*/
	rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(255), 2, 8, 0);
	putText(img_display, num, Point(matchLoc.x, matchLoc.y + templ.rows / 1.5), 1, 1, Scalar(255, 0, 255), 2);


	imshow("Source Image", img_display);
	imwrite(file_final, img_display);

	namedWindow("template");
	imshow("template", templ);
}
#endif