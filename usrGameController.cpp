#include "usrGameController.h"

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
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
		GAME_DOOR, GAME_INIT, GAME_IN} STATE;

Point* resultPoints;
resultpt *sortedResults;
double *similarityArr;
ofstream out("E:/qtdipdata.txt");
bool matchedFirstTime = 0, matchedSecondTime = 0;

void matchTemplate(int);
void matchTemplate(int, int);
void clipOriginPic(int mode);
void checkMatchedState();

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

	/*************����Ϊ����ӵĴ���******************/
	// ע�Ȿ����usrGameControl()��Լÿ���ӵ���20��
	counter++;
	int mode = 2;// 8*8=64ģʽ
				 
	////// ��Ƶ������ڸ�·����
	imwrite("frame.png", img);
	//string fname = "E:/frames/frame_";  
	//fname += ('0' + mode*mode / 10); fname += ('0' + mode*mode % 10); fname += '_';
	//fname += ('0' + counter / 100); fname += ('0' + counter / 10 % 10); fname += ('0' + counter % 10);
	//fname += (".png");
	//imwrite(fname, img);

	//clipOriginPic(mode);  //ֻдһ�Σ����������и���Сģ��
	
	// ��ס, img ���ǵ�ǰ֡������
	checkMatchedState();
	
	const int x0 = 52, y0 = 315, x1 = 489, y1 = 752;// Ŀ������������
	int d = (x1 - x0) / 4 / mode;

	
	double similarityAccuracy = 1e-1;
	if (STATE == GAME_INIT) {
		// ������Ϸ����״̬
		// ��һ��ƥ��
		
		
		if (!matchedFirstTime) {
			matchTemplate(mode);
			matchedFirstTime = true;
		}
		if (matchedFirstTime) {
			out << "successfully matched first time" << endl;
			for (int i = 0; i < mode*mode && similarityArr[i] < similarityAccuracy; ++i) {
				device->comHitDown();
				out << "successfully comHitDown" << endl;
				double scaleX, scaleY;
				scaleX = ((double)resultPoints[i].x + d * (1 / 2.0 + i%mode)) / pt.cols * 100;
				scaleY = ((double)resultPoints[i].y + d*(1 / 2.0 + i / mode)) / pt.rows * 100;
				out << "scaleX:" << scaleX << "scaleY" << scaleY << endl;
				device->comMoveToScale(scaleX, scaleY);
				/*device->comMoveTo(resultPoints[i].x + d * (1 / 2 + i%mode),
					resultPoints[i].y + d*(1 / 2 + i / mode));*/
				device->comHitUp();
			}
		}

		// �ڶ���ƥ�䣬���ǵ������Ѿ������˺ܶ࣬ƥ��ɹ���Ӧ�û���ߣ���������Ŀ�Դ�
		/*similarityAccuracy = 1e-10;
		if (matchedSecondTime) {
			for (int i = 0; i < mode*mode && sortedResults[i].similarity < similarityAccuracy; ++i) {
				device->comHitDown();
				device->comMoveToScale(((double)sortedResults[i].pt.x + argM.box.width) / pt.cols,
					((double)sortedResults[i].pt.y + argM.box.height) / pt.rows);
				device->comHitUp();
			}
		}
		else matchTemplate(mode), matchedSecondTime = 1;*/
	}
	//STATE = GAME_IN;

	

	return 0; 
}

//���ص�����
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

/*******segmentation***************/
void clipOriginPic(int mode) {
	//const int x0 = 92, y0 = 512, x1 = 992, y1 = 1412;// ������ֻ���ͼ�ı궨
	const int x0 = 52, y0 = 315, x1 = 489, y1 = 752;// ���screencapture
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
		cout << "�ɹ�д��img" << file_seg << endl;
	}
}

/****************ģ��ƥ��***************/
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

	double alpha = 2.5;
	resize(srcImage, img, Size(srcImage.cols / alpha, srcImage.rows / alpha));

	namedWindow("Source Image");

	Mat img_display;
	img.copyTo(img_display);
	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;
	result.create(result_rows, result_cols, CV_32FC1);
	match_method = TM_SQDIFF;

	resultPoints = new Point[mode*mode];
	similarityArr = new double[mode*mode];
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
		minVal = fabs(minVal);
		matchLoc = minLoc;
		similarityArr[i] = minVal;
		cout << "��" << i << "��ƴͼ�Ĳ���\nminValue=" << minVal << "\tminLoc=(" << minLoc.x << ", " << minLoc.y << ")\n";
		out << "��" << i << "��ƴͼ�Ĳ���\nminValue=" << minVal << "\tminLoc=(" << minLoc.x << ", " << minLoc.y << ")\n";

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

	// ������
	sortedResults = new resultpt[mode*mode];
	double val; int pos;

	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!�˴������㷨д����
	for (int i = 0; i < mode*mode; ++i) {
		val = similarityArr[i]; pos = 0;
		for (int j = 0; j !=i && j < mode*mode; ++j) {
			if (val < similarityArr[j]) pos++;
		}
		sortedResults[pos] = resultpt(resultPoints[i], i, val);
	}
	out << "sortedResults:\n";
	for (int i = 0; i < mode*mode; ++i) {
		out << i << ": " << sortedResults[i].similarity << endl;
	}
	out << "similarityArr:\n";
	for (int i = 0; i < mode*mode; ++i) {
		out << i << ": " << similarityArr[i] << endl;
	}
	//���������ṩһ��ȫ�ֽӿ� resultPoints[mode*mode] ����������ƴͼ�ӿ������
	//���������ṩһ��ȫ�ֽӿ� sortedResults[mode*mode] �������ƶ�������ƴͼ�ӿ�����
	//�������ȫ��ȷ��������ȡ���ƶ���ߵ���ͼ���ƶ������𲽵���
}

//ģ��ƥ��ӿ�2
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

	double alpha = 2.5;
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
	minVal = fabs(minVal);
	matchLoc = minLoc;
	cout << "��" << i << "��ƴͼ�Ĳ���\nminValue=" << minVal << "\tminLoc=(" << minLoc.x << ", " << minLoc.y << ")\n";

	/*resultPoints[i] = Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows);*/
	rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(255), 2, 8, 0);
	putText(img_display, num, Point(matchLoc.x, matchLoc.y + templ.rows / 1.5), 1, 1, Scalar(255, 0, 255), 2);


	imshow("Source Image", img_display);
	imwrite(file_final, img_display);

	namedWindow("template");
	imshow("template", templ);
}
void checkMatchedState() {
	Mat  templ, result, frame;
	frame = imread("frame.png");

	int mode;
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;

	string file_play("E:/play04.png"), file_seg("E:/seg00_small_00.png"), file_final("E:/final00.png"),
		file_door("E:/door02_02.png"), file_init("E:/init02_02.png");
	
	templ = imread(file_door);
	matchTemplate(frame, templ, result, TM_SQDIFF);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	minVal = fabs(minVal);
	qDebug() << "GAME_DOOR minVal: " << minVal << endl;
	out << "GAME_DOOR minVal: " << minVal << endl;
	if (minVal < 1e-8) STATE = GAME_DOOR, qDebug() << "GAME_DOOR matched success" << endl, out << "GAME_DOOR matched success" << endl;;

	templ = imread(file_init);
	matchTemplate(frame, templ, result, TM_SQDIFF);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	minVal = fabs(minVal);
	qDebug() << "GAME_INIT minVal: " << minVal << endl;
	out << "GAME_INIT minVal: " << minVal << endl;
	if (minVal < 1e-7) STATE = GAME_INIT, qDebug() << "GAME_INIT matched success" << endl, out << "GAME_INIT matched success" << endl;;
}
#endif