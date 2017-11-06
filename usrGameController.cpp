#include "usrGameController.h"

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <Windows.h>
using namespace std;
using namespace cv;

struct resultpt {
	Point pt;
	int index;
	double similarity;

	resultpt():pt(NULL){}
	resultpt(Point _pt, int _index, double _sim):
		pt(_pt), index(_index), similarity(_sim){}
};
enum {JIGTY, GAME_SELECT01, GAME_SELECT02,
		GAME_DOOR, GAME_INIT, GAME_IN, GAME_SUCCESS} STATE;

const int mode = 3;// 8*8=64模式
// (552, 1078)  clip (7,63) (547, 1023) ->real (540x960)
// square 893x893 in (1080x1920)
const int X0 = 98, Y0 = 518, X1 = 991, Y1 = 1411;
double d = (X1 - X0) / 2.0 / mode;

int success_game_in_cnt = 0;
double alpha = 2.5;
Point* resultPoints;
resultpt *matchedPics;
double *similarityArr;
ofstream out("E:/qtdipdata.txt");
Mat frame;
string  file_play("E:/_pic/play02_04.png"), file_seg("E:/_pic/seg02_00_00.png"), 
		file_final("E:/_pic/final02_00.png"), file_door("E:/_pic/door02_02.png"), 
		file_init("E:/_pic/init02_02.png"), file_in_square("E:/_pic/in_square02_00.png");
bool* finishedPic = new bool[mode*mode];
bool initFinishedPics(false);

void match_template(int mode);
void checkMatchedState(int mode);
bool checkSuccess(int mode);
void updateFilenames(int mode);
int diff_pics(Mat& src, Mat& tmp);

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
		device->comMoveToScale( ((double)argM.box.x + argM.box.width) / pt.cols,
								((double)argM.box.y + argM.box.height) / pt.rows);
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
	updateFilenames(mode);

	// 视频流存放在该路径下, 记住: frame 就是当前帧的数据
	// 要根据实际Total control窗口的大小调整, 此处为552x1078
	frame = img(Rect(7, 63, 540, 960));
	imwrite("frame.png", frame);
	
	// 先判别当前帧处于什么状态，开始游戏还是进行游戏
	checkMatchedState(mode); 
	
	bool isSuccess(false); 
	// 判别当前帧是否成功游戏，同时返回一个正确拼图子块的布尔索引
	isSuccess = checkSuccess(mode); 
	int moved_pics_cnt = 0;

	if (STATE == GAME_IN && !isSuccess) {
		// 进入游戏运行状态

		success_game_in_cnt++;
		out << "successfully matched " << success_game_in_cnt << " time" << endl;
		qDebug() << "successfully matched " << success_game_in_cnt << " time" << endl;

		for (int i = ((success_game_in_cnt-1)%mode)*mode; i < mode*mode; ++i) {
			if (finishedPic[i] == false) {
				out << "successfully comHitDown" << endl;
				double scaleX, scaleY;
				scaleX = (7 + matchedPics[i].pt.x * alpha) / pt.cols;
				scaleY = (63 - UP_CUT + matchedPics[i].pt.y * alpha) / pt.rows;
				out << "scaleX:" << scaleX << "scaleY" << scaleY << endl;
				qDebug() << "scaleX:" << scaleX << "scaleY" << scaleY << endl;
				device->comMoveToScale(scaleX, scaleY);
				device->comHitDown();

				Sleep(3000);
				scaleX = ( 7 + ((double)X0 + d * (1 + 2.0 * (matchedPics[i].index % mode))) / 2) / pt.cols;
				scaleY = (63 + ((double)Y0 + d * (1 + 2.0 * (matchedPics[i].index / mode))) / 2 - UP_CUT) / pt.rows;
				out << "scaleX:" << scaleX << "scaleY" << scaleY << endl;
				qDebug() << "\nscaleX:" << scaleX << "scaleY" << scaleY << endl;
				device->comMoveToScale(scaleX, scaleY);
				device->comHitUp();
				Sleep(3 * 1000);
				qDebug() << "successfully moved one pic" << endl;
				device->comMoveToScale(0, 0);//return original pos
				Sleep(3 * 1000);
				moved_pics_cnt++;
				if (moved_pics_cnt == 2*mode) break;
			}
		}
		//system("pause");
	}
	
	if (isSuccess) {
		qDebug() << " \n\n\nAll successfully matched, you can have fun debugging now!" << endl;
		device->comMoveToScale(0, 0);//return original pos
		system("pause");
	}

	return 0; 
}

// 鼠标回调函数
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

/****************模板匹配***************/
void match_template(int mode) {
	Mat src, seg, result, src_display;

	src = frame;
	resize(src, src, Size(src.cols / alpha, src.rows / alpha));
	src.copyTo(src_display);

	resultPoints = new Point[mode*mode];
	similarityArr = new double[mode*mode];
	string num = "00";	
	for (int i = 0; i < mode*mode; ++i) {
		num[0] = '0' + i / 10;
		num[1] = '0' + i % 10;
		seg.release(); result.release();
		file_seg[file_seg.length() - 6] = num[0];
		file_seg[file_seg.length() - 5] = num[1];
		seg = imread(file_seg, IMREAD_COLOR);
		resize(seg, seg, Size(seg.cols / alpha / 2, seg.rows / alpha / 2));

		matchTemplate(src, seg, result, TM_SQDIFF);
		normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
		double minVal; double maxVal; Point minLoc; Point maxLoc;
		Point matchLoc;
		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
		minVal = fabs(minVal);
		matchLoc = minLoc;
		similarityArr[i] = minVal;
		qDebug() << "第" << i << "个拼图的参数\nminValue=" << minVal << "\tminLoc=(" << minLoc.x << ", " << minLoc.y << ")\n";
		out << "第" << i << "个拼图的参数\nminValue=" << minVal << "\tminLoc=(" << minLoc.x << ", " << minLoc.y << ")\n";

		resultPoints[i] = Point(matchLoc.x + seg.cols / 2, matchLoc.y + seg.rows / 2);
		rectangle(src_display, matchLoc, Point(matchLoc.x + seg.cols, matchLoc.y + seg.rows), Scalar::all(255), 2, 8, 0);
		putText(src_display, num, Point(matchLoc.x, matchLoc.y + seg.rows / 1.5), 6, 1, Scalar(255, 0, 255), 2);
	}

	imwrite(file_final, src_display);
	namedWindow(file_final);
	imshow(file_final, src_display);

	// 排序结果
	matchedPics = new resultpt[mode*mode];
	double val; int pos;
	for (int i = 0; i < mode*mode; ++i) {
		val = similarityArr[i];
		matchedPics[i] = resultpt(resultPoints[i], i, val);
	}
	
	//本函数可提供一个全局接口 resultPoints[mode*mode] 即最终所有拼图子块的中心
	//本函数可提供一个全局接口 matchedPics[mode*mode] 即依相似度排序后的拼图子块中心
	//结果不完全正确，可以先取相似度最高的子图先移动，再逐步调整
}

/****************判断当前状态***************/
void checkMatchedState(int mode) {
	Mat  templ, result;
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;	
	
	frame = imread("frame.png");
	templ = imread(file_door);
	resize(templ, templ, Size(540, 960));
	templ = templ(Rect(1, 1, 500, 900));
	matchTemplate(frame, templ, result, TM_SQDIFF);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	minVal = fabs(minVal);
	qDebug() << "GAME_DOOR minVal: " << minVal << endl;
	out << "GAME_DOOR minVal: " << minVal << endl;
	if (minVal < 1e-8) STATE = GAME_DOOR, qDebug() << "GAME_DOOR matched success" << endl, out << "GAME_DOOR matched success" << endl;

	templ = imread(file_init);
	resize(templ, templ, Size(540, 960));
	templ = templ(Rect(0, 0, 500, 900));
	matchTemplate(frame, templ, result, TM_SQDIFF);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	minVal = fabs(minVal);
	qDebug() << "GAME_INIT minVal: " << minVal << endl;
	out << "GAME_INIT minVal: " << minVal << endl;
	if (minVal < 1e-9) STATE = GAME_INIT, qDebug() << "GAME_INIT matched success" << endl, out << "GAME_INIT matched success" << endl;

	Mat in_square(frame, Rect(Point(X0/2, Y0/2), Point(X1/2, Y1/2)));//551x1078 scale: x(0.11,0.90) y(0.30,0.70)
	Mat in_square_templ = imread(file_in_square);
	matchTemplate(in_square, in_square_templ, result, TM_SQDIFF);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	minVal = fabs(minVal);
	qDebug() << "GAME_IN minVal: " << minVal << endl;
	out << "GAME_IN minVal: " << minVal << endl;
	if (minVal < 1e-6) {
		STATE = GAME_IN,
		qDebug() << "GAME_IN matched success" << endl,
		out << "GAME_IN matched success" << endl;
	}
}

bool checkSuccess(int mode) {
	if (STATE != GAME_IN) return false;

	if (!initFinishedPics)
		for (int i = 0; i < mode*mode; ++i) finishedPic[i] = false;
	initFinishedPics = true;
	
	match_template(mode);
	Mat frame_copy;
	frame.copyTo(frame_copy);
	int xi0, yi0;
	double d_error;
	for (int i = 0; i < mode*mode; ++i) {
		xi0 = X0 + d * (0.5 + 2.0 * (matchedPics[i].index % mode));
		yi0 = Y0 + d * (0.5 + 2.0 * (matchedPics[i].index / mode));
		xi0 /= 2; yi0 /= 2;
		d_error = norm(resultPoints[i] - Point(xi0 / alpha, yi0 / alpha));
		cout << "d_error: " << d_error << endl;
		if (d_error < 80 / mode) {// 80/mode经验常数，可能需要调整
			finishedPic[i] = true;
			cout << i << " matched success" << endl;

			rectangle(frame_copy, Point(xi0, yi0), Point(xi0 + d / 2, yi0 + d / 2), Scalar::all(255), 2, 8, 0);
			string num = "00"; num[0] = '0' + i / 10; num[1] = '0' + i % 10;
			putText(frame_copy, num, Point(xi0, yi0 + d / 3), 2, 0.5, Scalar(255, 0, 255), 2);
		}
	}
	namedWindow("rst");
	imshow("rst", frame_copy);

	for (int i = 0; i < mode*mode; ++i)
		if (!finishedPic[i]) return false;

	cout << "ALL matched success" << endl;
	out << "ALL matched success" << endl;
	return true;
}

int diff_pics(Mat& src, Mat& tmp) {
	return cv::sum(cv::abs(src - tmp))[0];
}

void updateFilenames(int mode)
{
	// 写这个函数主要是因为我的模板图片的路径总是被我搬来搬去，不想换一次路径改一次了
	file_seg[file_seg.length() - 9] = mode*mode / 10 + '0';
	file_seg[file_seg.length() - 8] = mode*mode % 10 + '0';

	file_play[file_play.length() - 6] = mode / 10 + '0';
	file_play[file_play.length() - 5] = mode % 10 + '0';
	file_final[file_final.length() - 6] = mode / 10 + '0';
	file_final[file_final.length() - 5] = mode % 10 + '0';
	file_door[file_door.length() - 6] = mode / 10 + '0';
	file_door[file_door.length() - 5] = mode % 10 + '0';
	file_init[file_init.length() - 6] = mode / 10 + '0';
	file_init[file_init.length() - 5] = mode % 10 + '0';
	file_in_square[file_in_square.length() - 6] = mode / 10 + '0';
	file_in_square[file_in_square.length() - 5] = mode % 10 + '0';
}
#endif