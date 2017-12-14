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

	resultpt() :pt(NULL) {}
	resultpt(Point _pt, int _index, double _sim) :
		pt(_pt), index(_index), similarity(_sim) {}
};
enum {
	JIGTY, GAME_SELECT01, GAME_SELECT02,
	GAME_DOOR, GAME_INIT, GAME_IN, GAME_SUCCESS
} STATE;

const int mode = 5;// 8*8=64ģʽ
				   // (552, 1078)  clip (7,63) (547, 1023) ->real (540x960)
				   // square 893x893 in (1080x1920)
const int X0 = 98, Y0 = 518, X1 = 991, Y1 = 1411;
double d = (X1 - X0) / 2.0 / mode;
double matchValsCntToSeg[] = { 0, 0, 0, 0, 3e7, 6e6, 1e6, 1e6 };
double matchValsCntToNull[] = { 0, 0, 0, 0, 3e7, 6e6, 1e6, 1e6 };

bool second_in_main_func = false;
int success_game_in_cnt = 0;
int seg_cnt = 0;
int tmp_cnt;
int not_match_cnt = 0;
double alpha = 2.5;
double scaleX, scaleY;
double minVal; double maxVal;
Point minLoc; Point maxLoc; Point matchLoc;
Point* resultPoints = new Point[mode*mode];
Point rstLoc;
resultpt *matchedPics;
double *similarityArr;
ofstream out("E:/qtdipdata.txt");
Mat frame;
string  file_play("E:/_pic/play02_04.png"), file_seg("E:/_pic/seg02_00_00.png"),
file_final("E:/_pic/final02_00.png"), file_door("E:/_pic/door02_02.png"),
file_init("E:/_pic/init02_02.png"), file_in_square("E:/_pic/in_square02_00.png");
bool* finishedPics = new bool[mode*mode];
bool initFinishedPics(false);
string num = "00";
double matchVal;
vector<vector<Point> > contours_poly_up;
vector<vector<Point> > contours_poly_down;
vector<Rect> boundRectDown;
vector<Rect> boundRectUp;

void _match(Mat& src, Mat& seg, Mat& result, double& minVal, double& maxVal,
	Point& minLoc, Point& maxLoc, Point& matchLoc);
void find_seg_k_in_src(int seg_num, Point& rstLoc);
void find_seg_k_in_seg_all(int seg_cnt, int &tmp_cnt);
void checkMatchedState();
bool checkSuccess();
void updateFilenames(int seg_num);
double check_match(int& cnt, int);

#ifdef VIA_OPENCV
//�������ʼ��
usrGameController::usrGameController(void* qtCD)
{
	qDebug() << "usrGameController online.";
	device = new deviceCyberDip(qtCD);//�豸������
	cv::namedWindow(WIN_NAME);
	cv::setMouseCallback(WIN_NAME, mouseCallback, (void*)&(argM));
	counter = 0;

}

//����
usrGameController::~usrGameController()
{
	cv::destroyAllWindows();
	if (device != nullptr)
	{
		delete device;
	}
	qDebug() << "usrGameController offline.";
}

//����ͼ�� 
int usrGameController::usrProcessImage(cv::Mat& img)
{
	cv::Size imgSize(img.cols, img.rows - UP_CUT);
	if (imgSize.height <= 0 || imgSize.width <= 0)
	{
		qDebug() << "Invalid image. Size:" << imgSize.width << "x" << imgSize.height;
		return -1;
	}

	//��ȡͼ���Ե
	cv::Mat pt = img(cv::Rect(0, UP_CUT, imgSize.width, imgSize.height));
	cv::imshow(WIN_NAME, pt);

	//�ж�������ߴ�
	if (argM.box.x >= 0 && argM.box.x < imgSize.width&&
		argM.box.y >= 0 && argM.box.y < imgSize.height
		)
	{
		qDebug() << "X:" << argM.box.x << " Y:" << argM.box.y;
		if (argM.Hit)
		{
			device->comHitDown();
		}

		device->comMoveToScale(((double)argM.box.x + argM.box.width) / pt.cols,
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




	/*************����Ϊ����ӵĴ���******************/
	
	updateFilenames(mode);

	// ��Ƶ������ڸ�·����, ��ס: frame ���ǵ�ǰ֡������
	// Ҫ����ʵ��Total control���ڵĴ�С����, �˴�Ϊ552x1078
	frame = img(Rect(7, 63, 540, 960));
	cv::imwrite("frame.png", frame);
	
	// ���б�ǰ֡����ʲô״̬����ʼ��Ϸ���ǽ�����Ϸ
	checkMatchedState();

	if (STATE == GAME_IN) {
		// ������Ϸ����״̬
		qDebug() << "GAME_IN: successfully matched " << success_game_in_cnt << " time" << endl;
		
		matchVal = check_match(seg_cnt, seg_cnt);
		if (matchVal > matchValsCntToSeg[mode]) { // ˵�������д���

			matchVal = check_match(seg_cnt, -1);
			if (matchVal > matchValsCntToNull[mode]) { // ˵�������Ǳ������ͼ��ռ��

				find_seg_k_in_seg_all(seg_cnt, tmp_cnt);
				find_seg_k_in_src(tmp_cnt, rstLoc);

				scaleX = (7 + rstLoc.x * alpha) / pt.cols;
				scaleY = (63 - UP_CUT + rstLoc.y * alpha) / pt.rows;
				device->comMoveToScale(scaleX, scaleY);
				device->comHitDown();

				Sleep(2 * 1000);
				scaleX = (7 + ((double)X0 + d * (1 + 2.0 * (tmp_cnt % mode))) / 2) / pt.cols;
				scaleY = (63 + ((double)Y0 + d * (1 + 2.0 * (tmp_cnt / mode))) / 2 - UP_CUT) / pt.rows;
				device->comMoveToScale(scaleX, scaleY);
				device->comHitUp();

				device->comMoveToScale(0, 0);//return original pos
				Sleep(3 * 1000);
				
			}

			else { // ˵�����ﻹ�ǿյ�

				// ƥ��һ��seg_cnt, �ҵ�ƥ��λ��rstLoc
				find_seg_k_in_src(seg_cnt, rstLoc);

				scaleX = (7 + rstLoc.x * alpha) / pt.cols;
				scaleY = (63 - UP_CUT + rstLoc.y * alpha) / pt.rows;
				device->comMoveToScale(scaleX, scaleY);
				device->comHitDown();

				Sleep(2 * 1000);
				scaleX = (7 + ((double)X0 + d * (1 + 2.0 * (seg_cnt % mode))) / 2) / pt.cols;
				scaleY = (63 + ((double)Y0 + d * (1 + 2.0 * (seg_cnt / mode))) / 2 - UP_CUT) / pt.rows;
				device->comMoveToScale(scaleX, scaleY);
				device->comHitUp();

				device->comMoveToScale(0, 0);//return original pos
				Sleep(3 * 1000);
			}
			not_match_cnt++;
		}
		else { // ˵������û�д������Ǽ���¼һ�³ɹ��ӿ���
			finishedPics[seg_cnt] = true;
			seg_cnt++; seg_cnt %= mode * mode;
		}

		if (not_match_cnt == 2) { // ����һֱ��not_match���Ǽ�������֮������ѭ��
			seg_cnt++; seg_cnt %= mode * mode;
			not_match_cnt = 0;
		}

		// ��������ǲ��ǳɹ���������
		if ( checkSuccess() )  
			qDebug() << "ALL matched success" << endl,
			system("pause");
	}

	return 0;
}

// ���ص�����
void mouseCallback(int event, int x, int y, int flags, void*param)
{
	usrGameController::MouseArgs* m_arg = (usrGameController::MouseArgs*)param;
	switch (event)
	{
	case CV_EVENT_MOUSEMOVE: // ����ƶ�ʱ
	{
		if (m_arg->Drawing)
		{
			m_arg->box.width = x - m_arg->box.x;
			m_arg->box.height = y - m_arg->box.y;
		}
	}
	break;
	case CV_EVENT_LBUTTONDOWN:case CV_EVENT_RBUTTONDOWN: // ��/�Ҽ�����
	{
		m_arg->Hit = event == CV_EVENT_RBUTTONDOWN;
		m_arg->Drawing = true;
		m_arg->box = cvRect(x, y, 0, 0);
	}
	break;
	case CV_EVENT_LBUTTONUP:case CV_EVENT_RBUTTONUP: // ��/�Ҽ�����
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

/****************ģ��ƥ��***************/
void find_seg_k_in_seg_all(int seg_cnt, int &tmp_cnt) {
	// ����seg_cnt��λ�ã�������
	Mat src, result, seg;
	frame = imread("frame.png");

	double xi0, yi0;
	xi0 = X0 + d * (1 + 2.0 * (seg_cnt % mode));
	yi0 = Y0 + d * (1 + 2.0 * (seg_cnt / mode));
	xi0 /= 2; yi0 /= 2;

	double dSize = d;
	src = frame(Rect(Point(xi0 - dSize / 2, yi0 - dSize / 2), Size(dSize, dSize)));

	// ȫ���ӿ鶼�Ž���ƥ�䣬�Ƚ�˭�����ƶ����

	double* minValues = new double[mode*mode];
	for (int seg_num = 0; seg_num < mode*mode; ++seg_num) {
		// �г�seg
		updateFilenames(seg_num);
		seg = imread(file_seg);
		resize(seg, seg, Size(seg.cols / 2, seg.rows / 2));

		// ƥ��src��seg
		_match(src, seg, result, minVal, maxVal, minLoc, maxLoc, matchLoc);

		// ����minVal
		minValues[seg_num] = minVal;
	}

	// �ҵ�����ֵ
	double maxNum = *std::min_element(minValues, minValues + mode * mode);
	tmp_cnt = std::find(minValues, minValues + mode * mode, maxNum) - minValues;
	qDebug() << "tmp_cnt = " << tmp_cnt << endl;
}

void find_seg_k_in_src(int seg_num, Point& rstLoc) {
	// �г�src
	Mat src, seg, result, src_display;
	src = frame;
	src.copyTo(src_display);
	resize(src, src, Size(src.cols / alpha, src.rows / alpha));

	// �г�seg
	updateFilenames(seg_num);
	seg = imread(file_seg);
	resize(seg, seg, Size(seg.cols / alpha / 2, seg.rows / alpha / 2));

	// ƥ��src��seg
	_match(src, seg, result, minVal, maxVal, minLoc, maxLoc, matchLoc);

	// ����ƥ����rstLoc
	rstLoc = Point(matchLoc.x + seg.cols / 2, matchLoc.y + seg.rows / 2);
	resultPoints[seg_num] = rstLoc;
	rectangle(src_display, matchLoc, Point(matchLoc.x + seg.cols, matchLoc.y + seg.rows), Scalar::all(255), 2, 8, 0);
	putText(src_display, num, Point(matchLoc.x, matchLoc.y + seg.rows / 1.5), 6, 1, Scalar(255, 0, 255), 2);

	// ��ʾƥ����
	cv::imwrite(file_final, src_display);
	namedWindow(file_final);
	imshow(file_final, src_display);
}

void _match(Mat& src, Mat& seg, Mat& result, double& minVal, double& maxVal,
	Point& minLoc, Point& maxLoc, Point& matchLoc) {

	matchTemplate(src, seg, result, TM_SQDIFF);
	//normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	matchLoc = minLoc;

}


/****************�жϵ�ǰ״̬***************/
double check_match(int& cnt, int pic_num) {
	// �����жϵ�cnt��������ͼƬpic�Ƿ�����
	// pic_num = -1 ʱ��ʾ��defaultƥ�䣻������ʾ���ӿ�seg[pic_num]ƥ��
	
	// ��cntLoc�����г�����Ϊsrc
	Mat src, seg, result;
	frame = imread("frame.png");

	double xi0, yi0;
	xi0 = X0 + d * (1 + 2.0 * (seg_cnt % mode));
	yi0 = Y0 + d * (1 + 2.0 * (seg_cnt / mode));

	double dSize = d;
	src = frame(Rect(Point(xi0 / 2 - dSize / 2, yi0 / 2 - dSize / 2), Size(dSize, dSize)));
	
	// �г�seg
	updateFilenames(pic_num);
	if (pic_num == -1) {
		seg = imread(file_play); 
		seg = seg(Rect(Point(xi0 - dSize / 2, yi0 - dSize / 2), Size(dSize, dSize)));
	}
	else 
		seg = imread(file_seg);

	resize(seg, seg, Size(seg.cols / 2, seg.rows / 2));

	// ƥ��src��seg
	_match(src, seg, result, minVal, maxVal, minLoc, maxLoc, matchLoc);

	return minVal;
}

void checkMatchedState() {

	Mat  templ, result;
	frame = imread("frame.png");
	templ = imread(file_door);
	resize(templ, templ, Size(540, 960));
	templ = templ(Rect(1, 1, 500, 900));
	_match(frame, templ, result, minVal, maxVal, minLoc, maxLoc, matchLoc);
	qDebug() << "GAME_DOOR minVal: " << minVal << endl;
	if (minVal < 1e8) STATE = GAME_DOOR, qDebug() << "GAME_DOOR matched success" << endl;

	templ = imread(file_init);
	resize(templ, templ, Size(540, 960));
	templ = templ(Rect(0, 0, 500, 900));
	_match(frame, templ, result, minVal, maxVal, minLoc, maxLoc, matchLoc);
	qDebug() << "GAME_INIT minVal: " << minVal << endl;
	if (minVal < 1e9) STATE = GAME_INIT, qDebug() << "GAME_INIT matched success" << endl;

	Mat in_square(frame, Rect(Point(X0 / 2, Y0 / 2), Point(X1 / 2, Y1 / 2)));//551x1078 scale: x(0.11,0.90) y(0.30,0.70)
	Mat in_square_templ = imread(file_in_square);
	_match(in_square, in_square_templ, result, minVal, maxVal, minLoc, maxLoc, matchLoc);
	qDebug() << "GAME_IN minVal: " << minVal << endl;
	if (minVal < 6e9) {
		STATE = GAME_IN,
			qDebug() << "GAME_IN matched success" << endl;
	}

	// ��ʼ��finishedPics[N]
	if (!initFinishedPics)
		for (int i = 0; i < mode*mode; ++i) finishedPics[i] = false;
	initFinishedPics = true;
}

bool checkSuccess() {

	if (STATE != GAME_IN) return false;

	for (int i = 0; i < mode*mode; ++i)
		if (!finishedPics[i]) return false;

	return true;
}

void updateFilenames(int seg_num = 0)
{
	// д���������Ҫ����Ϊ�ҵ�ģ��ͼƬ��·�����Ǳ��Ұ�����ȥ�����뻻һ��·����һ����
	file_seg[file_seg.length() - 9] = mode*mode / 10 + '0';
	file_seg[file_seg.length() - 8] = mode*mode % 10 + '0';
	file_seg[file_seg.length() - 6] = seg_num / 10 + '0';
	file_seg[file_seg.length() - 5] = seg_num % 10 + '0';

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