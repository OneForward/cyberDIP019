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

const int mode = 4;// 8*8=64ģʽ
// (552, 1078)  clip (7,63) (547, 1023) ->real (540x960)
// square 893x893 in (1080x1920)
const int X0 = 98, Y0 = 518, X1 = 991, Y1 = 1411;
double d = (X1 - X0) / 2.0 / mode;

int success_game_in_cnt = 0;
int seg_cnt=0;
double alpha = 2.5;
double scaleX, scaleY;
Point* resultPoints;
Point rstPoint;
resultpt *matchedPics;
double *similarityArr;
ofstream out("E:/qtdipdata.txt");
Mat frame;
string  file_play("E:/_pic/play02_04.png"), file_seg("E:/_pic/seg02_00_00.png"), 
		file_final("E:/_pic/final02_00.png"), file_door("E:/_pic/door02_02.png"), 
		file_init("E:/_pic/init02_02.png"), file_in_square("E:/_pic/in_square02_00.png");
bool* finishedPic = new bool[mode*mode];
bool initFinishedPics(false);

vector<vector<Point> > contours_poly_up;
vector<vector<Point> > contours_poly_down;
vector<Rect> boundRectDown;
vector<Rect> boundRectUp;

void match_template(int mode);
void match_template_1_pic_to_1_pic(int mode, int seg_num);
void match_template_1_pic_to_multi_pic(int mode, int seg_cnt, int &tmp_cnt);
void checkMatchedState(int mode);
bool checkSuccess(int mode);
bool checkDownMargin();
bool checkUpMargin();
void updateFilenames(int mode);
int diff_pics(Mat& src, Mat& tmp);


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
		qDebug() << "Invalid image. Size:" << imgSize.width <<"x"<<imgSize.height;
		return -1;
	}

	//��ȡͼ���Ե
	cv::Mat pt = img(cv::Rect(0, UP_CUT, imgSize.width,imgSize.height));
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

	/*************����Ϊ����ӵĴ���******************/
	updateFilenames(mode);

	// ��Ƶ������ڸ�·����, ��ס: frame ���ǵ�ǰ֡������
	// Ҫ����ʵ��Total control���ڵĴ�С����, �˴�Ϊ552x1078
	frame = img(Rect(7, 63, 540, 960));
	cv::imwrite("frame.png", frame);
	
	// ���б�ǰ֡����ʲô״̬����ʼ��Ϸ���ǽ�����Ϸ
	checkMatchedState(mode); 
	
	bool isSuccess(false); 
	// �б�ǰ֡�Ƿ�ɹ���Ϸ��ͬʱ����һ����ȷƴͼ�ӿ�Ĳ�������
	if (seg_cnt==16)
		isSuccess = checkSuccess(mode); 
	int moved_seg_cnt = 0;
	
	//������һ�εĻ�е���ƶ����Ա�鿴ƥ����
	if (success_game_in_cnt == 0) STATE = GAME_DOOR, success_game_in_cnt++;

	if (STATE == GAME_IN && !isSuccess) {
		// ������Ϸ����״̬

		out << "successfully matched " << success_game_in_cnt << " time" << endl;
		qDebug() << "successfully matched " << success_game_in_cnt << " time" << endl;

		// ������Ҫ�ƶ���cnt��seg
		seg_cnt %= 16;
		if (finishedPic[seg_cnt] == false) {
			
			scaleX = (7 + rstPoint.x * alpha) / pt.cols;
			scaleY = (63 - UP_CUT + rstPoint.y * alpha) / pt.rows;
			device->comMoveToScale(scaleX, scaleY);
			device->comHitDown();

			Sleep(2 * 1000);
			scaleX = ( 7 + ((double)X0 + d * (1 + 2.0 * (seg_cnt % mode))) / 2) / pt.cols;
			scaleY = (63 + ((double)Y0 + d * (1 + 2.0 * (seg_cnt / mode))) / 2 - UP_CUT) / pt.rows;
			device->comMoveToScale(scaleX, scaleY);
			device->comHitUp();
			Sleep(2 * 1000);
			
			qDebug() << "successfully moved matched pic" << endl;
			device->comMoveToScale(0, 0);//return original pos
			Sleep(3 * 1000);
		}
		
		// ����img�ǲ���ʵʱ���µ�
		cv::imwrite("frame_after_translated.png", img);
		// cv::imwrite("frame.png", frame);

		// ע����ʵ����ֻ�ǰ�ƥ�䵽������ͼ�ƶ�����cnt��λ�ã����Ǹ�ͼƬ���ܲ��������ı��cnt���ӿ�Ŷ
		// ��������������һ�δ��룬��ԭ��ƥ�䲢�ƶ���ͼƬ
		int tmp_cnt;
		match_template_1_pic_to_multi_pic(mode, seg_cnt, tmp_cnt);

		if (tmp_cnt != seg_cnt) {
			
			scaleX = ( 7 + ((double)X0 + d * (1 + 2.0 * (tmp_cnt % mode))) / 2) / pt.cols;
			scaleY = (63 + ((double)Y0 + d * (1 + 2.0 * (tmp_cnt / mode))) / 2 - UP_CUT) / pt.rows;
			device->comMoveToScale(scaleX, scaleY);
			device->comHitDown();

			Sleep(2 * 1000);
			scaleX = ( 7 + ((double)X0 + d * (1 + 2.0 * (seg_cnt % mode))) / 2) / pt.cols;
			scaleY = (63 + ((double)Y0 + d * (1 + 2.0 * (seg_cnt / mode))) / 2 - UP_CUT) / pt.rows;
			device->comMoveToScale(scaleX, scaleY);
			device->comHitUp();
			Sleep(2 * 1000);
			
			qDebug() << "successfully moved matched pic" << endl;
			device->comMoveToScale(0, 0);//return original pos
			Sleep(3 * 1000);
		}
		else
			seg_cnt++;


		int remainedPics = 0;
		for (int i = 0; i < mode*mode; ++i) {
			if (finishedPic[i] == false) remainedPics++;
		}
		

	}
	
	
	if (isSuccess) {
		qDebug() << " \n\n\nAll successfully matched, you can have fun debugging now!" << endl;
		device->comMoveToScale(0, 0);//return original pos
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
void match_template_1_pic_to_multi_pic(int mode, int seg_cnt, int &tmp_cnt){
	// ����seg_cnt��λ�ã�������
	Mat src, result, seg;
	frame = imread("frame.png");
	
	double xi0, yi0;
	xi0 = X0 + d * (1 + 2.0 * (seg_cnt % mode));
	yi0 = Y0 + d * (1 + 2.0 * (seg_cnt / mode));
	xi0 /= 2; yi0 /= 2;

	src = frame( Rect( Point(xi0, yi0), Size(d*1.5/2, d*1.5/2) ) );

	// ȫ���ӿ鶼�Ž���ƥ�䣬�Ƚ�˭�����ƶ����
	string num = "00";
	double* minValues = new double[mode*mode];
	for (int seg_num = 0; seg_num < mode*mode; ++seg_num) {
		// �г�seg
		num[0] = '0' + seg_num / 10;
		num[1] = '0' + seg_num % 10;
		file_seg[file_seg.length() - 6] = num[0];
		file_seg[file_seg.length() - 5] = num[1];
		seg = imread(file_seg);
		resize(seg, seg, Size(seg.cols / alpha / 2, seg.rows / alpha / 2));

		// ƥ�� seg �� src
		matchTemplate(src, seg, result, TM_SQDIFF);
		normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
		double minVal; double maxVal; Point minLoc; Point maxLoc; Point matchLoc;
		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
		minVal = fabs(minVal);
		matchLoc = minLoc;

		// ����minVal
		minValues[seg_num] = minVal;
	}
	
	// �ҵ�����ֵ
	double maxNum = *std::max_element(minValues, minValues + mode * mode);
  	tmp_cnt = std::find(minValues, minValues + mode * mode, maxNum) - minValues;
}

void match_template_1_pic_to_1_pic(int mode, int seg_num) {
	// �г�src
	Mat src, seg, result, src_display;
	src = frame;
	resize(src, src, Size(src.cols / alpha, src.rows / alpha));
	src.copyTo(src_display);

	// �г�seg
	string num = "00";	
	num[0] = '0' + seg_num / 10;
	num[1] = '0' + seg_num % 10;
	file_seg[file_seg.length() - 6] = num[0];
	file_seg[file_seg.length() - 5] = num[1];
	seg = imread(file_seg);
	resize(seg, seg, Size(seg.cols / alpha / 2, seg.rows / alpha / 2));

	// ƥ��src��seg
	matchTemplate(src, seg, result, TM_SQDIFF);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
	double minVal; double maxVal; Point minLoc; Point maxLoc; Point matchLoc;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	minVal = fabs(minVal);
	matchLoc = minLoc;

	// ����ƥ����rstPoint
	resultPoints[seg_num] = Point(matchLoc.x + seg.cols / 2, matchLoc.y + seg.rows / 2);
	rectangle(src_display, matchLoc, Point(matchLoc.x + seg.cols, matchLoc.y + seg.rows), Scalar::all(255), 2, 8, 0);
	putText(src_display, num, Point(matchLoc.x, matchLoc.y + seg.rows / 1.5), 6, 1, Scalar(255, 0, 255), 2);

	// ��ʾƥ����
	cv::imwrite(file_final, src_display);
	namedWindow(file_final);
	imshow(file_final, src_display);
}


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
		qDebug() << "��" << i << "��ƴͼ�Ĳ���\nminValue=" << minVal << "\tminLoc=(" << minLoc.x << ", " << minLoc.y << ")\n";
		out << "��" << i << "��ƴͼ�Ĳ���\nminValue=" << minVal << "\tminLoc=(" << minLoc.x << ", " << minLoc.y << ")\n";

		resultPoints[i] = Point(matchLoc.x + seg.cols / 2, matchLoc.y + seg.rows / 2);
		rectangle(src_display, matchLoc, Point(matchLoc.x + seg.cols, matchLoc.y + seg.rows), Scalar::all(255), 2, 8, 0);
		putText(src_display, num, Point(matchLoc.x, matchLoc.y + seg.rows / 1.5), 6, 1, Scalar(255, 0, 255), 2);
	}

	cv::imwrite(file_final, src_display);
	namedWindow(file_final);
	imshow(file_final, src_display);

	// ������
	matchedPics = new resultpt[mode*mode];
	double val; int pos;
	for (int i = 0; i < mode*mode; ++i) {
		val = similarityArr[i];
		matchedPics[i] = resultpt(resultPoints[i], i, val);
	}
	
	//���������ṩһ��ȫ�ֽӿ� resultPoints[mode*mode] ����������ƴͼ�ӿ������
	//���������ṩһ��ȫ�ֽӿ� matchedPics[mode*mode] �������ƶ�������ƴͼ�ӿ�����
	//�������ȫ��ȷ��������ȡ���ƶ���ߵ���ͼ���ƶ������𲽵���
}

/****************�жϵ�ǰ״̬***************/
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
		if (d_error < 80 / mode) {// 80/mode���鳣����������Ҫ����
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

bool checkDownMargin() {
	// �±�Ե�ļ��
	Mat A = imread(file_final), AA = imread(file_init);
	resize(AA, AA, Size(216, 384));
	Mat C = cv::abs(A - AA);
	Mat DownPic = C(Rect(Point(0, 1411 / 5), Point(216, 384)));
	cvtColor(DownPic, DownPic, cv::COLOR_RGB2GRAY);
	threshold(DownPic, DownPic, 10, 255, THRESH_BINARY);
	erode(DownPic, DownPic, cv::getStructuringElement(MORPH_RECT, Size(5, 5)));

	Mat threshold_output = DownPic;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly_down(contours.size());
	vector<Rect> boundRectDown(contours.size());

	if (contours.size() == 0) return false;
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly_down[i], 3, true);
		boundRectDown[i] = boundingRect(Mat(contours_poly_down[i]));
	}
	
	return true;
}

bool checkUpMargin() {
	// �ϱ�Ե�ļ��
	Mat A = imread(file_final), AA = imread(file_init);
	resize(AA, AA, Size(216, 384));
	Mat C = cv::abs(A - AA);
	Mat UpPic = C(Rect(Point(0, 0), Point(216, 518 / 5)));
	cvtColor(UpPic, UpPic, cv::COLOR_RGB2GRAY);
	threshold(UpPic, UpPic, 10, 255, THRESH_BINARY);
	erode(UpPic, UpPic, cv::getStructuringElement(MORPH_RECT, Size(5, 5)));

	Mat threshold_output = UpPic;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly_up(contours.size());
	vector<Rect> boundRectUp(contours.size());

	if (contours.size() == 0) return false;
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly_up[i], 3, true);
		boundRectUp[i] = boundingRect(Mat(contours_poly_up[i]));
	}

	return true;
}

void updateFilenames(int mode)
{
	// д���������Ҫ����Ϊ�ҵ�ģ��ͼƬ��·�����Ǳ��Ұ�����ȥ�����뻻һ��·����һ����
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