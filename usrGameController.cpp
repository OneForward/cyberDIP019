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
	GAME_SELECT01, GAME_SELECT02, GAME_SELECT03,
	GAME_SELECT04, GAME_IN, GAME_STOP, GAME_SUCCESS
} STATE;

enum { CHECK_ALL, CHECK_UP_MARGIN, CHECK_BOTTOM_MARGIN };

const int mode = 8;// 8*8=64模式
				   // (552, 1078)  clip (7,63) (547, 1023) ->real (540x960)
				   // square 893x893 in (1080x1920)
const int X0 = 98, Y0 = 518, X1 = 991, Y1 = 1411;
const int STOP_X = 1018, STOP_Y = 64, STOP_R = 87;
Point STOP = Point(STOP_X, STOP_Y);
double d = (X1 - X0) / 2.0 / mode;
double matchValsCntToSeg[] = { 0, 0, 5e7, 2e7, 3e7, 6e6, 6e6, 8e6, 4e6 };
double matchValsCntToNull[] = { 0, 0, 3e7, 3e7, 3e7, 6e6, 6e6, 1e6, 1e6 };

int success_game_in_cnt = 0;
int seg_cnt = 0;
int tmp_cnt;
int not_match_cnt = 0;
int remainedPics = mode * mode;

double alpha = 2.5;
double scaleX, scaleY;
double minVal; double maxVal;
Point minLoc, maxLoc, matchLoc, rstLoc, endLoc, fromLoc, pos;
ofstream out("E:/qtdipdata.txt");
string	game_select01("E:/_pic/GAME_SELECT01.png"), game_select02("E:/_pic/GAME_SELECT02.png"),
game_select03("E:/_pic/GAME_SELECT03.png"), game_select04("E:/_pic/GAME_SELECT04.png"),
game_stop("E:/_pic/GAME_STOP.png"), game_stop_symbol("E:/_pic/stop.png"),
game_success01("E:/_pic/GAME_SUCCESS01.png"), game_success02("E:/_pic/GAME_SUCCESS02.png");;

string  file_play("E:/_pic/play02_04.png"), file_seg("E:/_pic/seg02_00_00.png"),
file_final("E:/_pic/final02_00.png"), file_door("E:/_pic/door02_02.png"),
file_init("E:/_pic/init02_02.png"), file_in_square("E:/_pic/in_square02_00.png");
bool* finishedPics = new bool[mode*mode];
bool initFinishedPics(false);
string num = "00";
double matchVal;
Mat frame;

void _match(Mat& src, Mat& seg, Mat& result, double& minVal, double& maxVal,
	Point& minLoc, Point& maxLoc, Point& matchLoc);
void find_where_is_seg_k(int seg_num, Point& rstLoc, int CHECK_TYPE = CHECK_ALL);
void find_who_is_at_pos(Point& pos, int &seg_cnt);
void find_accurate_pos_at_pos(Point& pos, int &seg_cnt, Point& rstLoc);
bool checkSuccess();
bool checkMargin(int);
void updateFilenames(int seg_num);
double check_match(int& cnt, int);

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
		qDebug() << "Invalid image. Size:" << imgSize.width << "x" << imgSize.height;
		return -1;
	}

	//截取图像边缘
	cv::Mat pt = img(cv::Rect(0, UP_CUT, imgSize.width, imgSize.height));
	cv::imshow(WIN_NAME, pt);

	//判断鼠标点击尺寸
	if (argM.box.x >= 0 && argM.box.x < imgSize.width&&
		argM.box.y >= 0 && argM.box.y < imgSize.height
		)
	{
		qDebug() << "X:" << argM.box.x << " Y:" << argM.box.y;
		
		cout << "argM.box" << argM.box << endl;
		cout << "argM.Hit" << argM.Hit << endl;
		cout << "argM.Drawing" << argM.Drawing << endl;
		device->comMoveToScale(((double)argM.box.x + argM.box.width) / pt.cols,
			((double)argM.box.y + argM.box.height) / pt.rows);
		
		argM.box.x = -1; argM.box.y = -1;
		
		if (argM.Hit)
		{
			device->comHitDown();
		}
		else
		{
			device->comHitOnce(); 
		}
	}

	
	//move_from_to(Point(332-7, 543), Point(339-7, 882), pt);
	/*imwrite("E:/_pic/pt.png", pt);
	imwrite("pt.png", pt);
	system("pause");*/

	/*************以下为我添加的代码******************/
	cout << "当前拼图模式为 mode = " << mode << endl;
	updateFilenames(mode);

	// 视频流存放在该路径下, 记住: frame 就是当前帧的数据
	// 要根据实际Total control窗口的大小调整, 此处为552x1078
	frame = img(Rect(7, 65, 540, 960));
	cv::imwrite("frame.png", frame);
	////system("pause");
	// 先判别当前帧处于什么状态，开始游戏还是进行游戏
	checkFrameState(pt);
	//STATE = GAME_IN;
	// 检查我们是不是成功啦啦啦啦
	if (checkSuccess())
		cout << "ALL matched success" << endl,
		system("pause");

	if (STATE == GAME_IN) {
		// 进入游戏运行状态
		cout << "GAME_IN: 成功进入游戏运行状态 " << endl;

		matchVal = check_match(seg_cnt, seg_cnt);
		cout << " 目前考察第 cnt =  " << seg_cnt << " 个区域的子块\tmatchValToSeg = " << matchVal << endl;
		if (matchVal > matchValsCntToSeg[mode]) { // 说明这里有错误

			cout << " 这里有错误 " << endl;
			matchVal = check_match(seg_cnt, -1);
			cout << " cnt =  " << seg_cnt << "\tmatchValToNull = " << matchVal << endl;

			if (matchVal > matchValsCntToNull[mode]) { // 说明这里是被别的子图误占了

				cout << " 这里是被别的子图误占了 " << endl;
				pos = Point((X0 + d * (1 + 2.0 * (seg_cnt % mode))) / 2.0,
					(Y0 + d * (1 + 2.0 * (seg_cnt / mode))) / 2.0);
				find_who_is_at_pos(pos, tmp_cnt);
				find_accurate_pos_at_pos(pos, tmp_cnt, fromLoc);
				cout << "起始点位置坐标fromLoc: " << fromLoc << endl << endl;

				endLoc = Point((X0 + d * (1 + 2.0 * (tmp_cnt % mode))) / 2.0,
					(Y0 + d * (1 + 2.0 * (tmp_cnt / mode))) / 2.0);
				move_from_to(fromLoc, endLoc, pt);

			}

			else { // 说明这里还是空的

				cout << " 这里这里还是空的 " << endl;
				// 匹配一下seg_cnt, 找到匹配位置rstLoc
				find_where_is_seg_k(seg_cnt, fromLoc);
				cout << "初始位置坐标fromLoc: " << fromLoc << endl << endl;

				endLoc = Point((X0 + d * (1 + 2.0 * (seg_cnt % mode))) / 2.0,
					(Y0 + d * (1 + 2.0 * (seg_cnt / mode))) / 2.0);
				move_from_to(fromLoc, endLoc, pt);

			}
			not_match_cnt++;
		}
		else { // 说明这里没有错误，我们记录一下成功子块标号

			cout << " 这里没有错误，我们记录一下成功子块标号 " << endl;
			finishedPics[seg_cnt] = true;
			seg_cnt++; seg_cnt %= mode * mode; not_match_cnt = 0;
		}

		if (not_match_cnt == 2) { // 避免一直在not_match的那几个区域之间陷入循环

			cout << " 避免一直在not_match的那几个区域之间陷入循环 " << endl;
			seg_cnt++; seg_cnt %= mode * mode;
			not_match_cnt = 0;
		}
		cout << "\nremainedPics = " << remainedPics << endl << endl;
		if (remainedPics <= mode) { // 当余下的图片较少时，开始检查上下边缘

			cout << " 当余下的图片较少时，开始检查上下边缘 " << endl;
			if (checkMargin(CHECK_UP_MARGIN)) { // 移动上边缘子块

				cout << " 移动上边缘子块 " << endl;
				cout << "初始位置坐标rstLoc: " << rstLoc << endl << endl;
				endLoc = Point(540 / 2, Y0 / 2 / 2);
				if (abs(rstLoc.x - endLoc.x) + abs(rstLoc.y - endLoc.y) > 100)
					move_from_to(rstLoc, endLoc, pt);

				else {

					cout << " 将在上边缘中间等待的子块移至该去的位置 " << endl;
					find_who_is_at_pos(rstLoc, tmp_cnt);
					find_accurate_pos_at_pos(rstLoc, tmp_cnt, fromLoc);
					cout << "初始位置坐标rstLoc: " << fromLoc << endl << endl;

					endLoc = Point((X0 + d * (1 + 2.0 * (tmp_cnt % mode))) / 2.0,
						(Y0 + d * (1 + 2.0 * (tmp_cnt / mode))) / 2.0);
					move_from_to(fromLoc, endLoc, pt);
				}
			}
			if (checkMargin(CHECK_BOTTOM_MARGIN)) { //  移动下边缘子块

				cout << " 移动下边缘子块 " << endl;
				cout << "初始位置坐标rstLoc: " << rstLoc << endl << endl;
				endLoc = Point(540 / 2, (1920 + Y1) / 2 / 2);
				if (abs(rstLoc.x - endLoc.x) + abs(rstLoc.y - endLoc.y) > 100)
					move_from_to(rstLoc, endLoc, pt);

				else {

					cout << " 将在下边缘中间等待的子块移至该去的位置 " << endl;
					find_who_is_at_pos(rstLoc, tmp_cnt);
					find_accurate_pos_at_pos(rstLoc, tmp_cnt, fromLoc);
					cout << "初始位置坐标rstLoc: " << fromLoc << endl << endl;

					endLoc = Point((X0 + d * (1 + 2.0 * (tmp_cnt % mode))) / 2.0,
						(Y0 + d * (1 + 2.0 * (tmp_cnt / mode))) / 2.0);
					move_from_to(fromLoc, endLoc, pt);
				}
			}

		}

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
		}break;
	}
	
	case CV_EVENT_LBUTTONDOWN:case CV_EVENT_RBUTTONDOWN: // 左/右键按下
	{
		m_arg->Hit = event == CV_EVENT_RBUTTONDOWN;
		m_arg->Drawing = true;
		m_arg->box = cvRect(x, y, 0, 0);
		break;
	}
	
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
		break;
	}
	
	}
}

/*********************操作机械臂*********************/
void usrGameController::move_from_to(cv::Point& fromLoc, cv::Point& toLoc, cv::Mat& pt) {
	
	if (fromLoc.x < 0 || fromLoc.y < 0 || fromLoc.x > 540 || fromLoc.y > 960 || norm(fromLoc - STOP / 2) < 87 / 2 ||
		toLoc.x < 0 || toLoc.y < 0 || toLoc.x > 540 || toLoc.y > 960 || norm(toLoc - STOP / 2) < 87 / 2) {
			qDebug() << "\nwarning pos out of bound!\n"; return; 
	}
	double t = 1.5;

	scaleX = (7.0 + fromLoc.x) / pt.cols;
	scaleY = ((double)fromLoc.y) / pt.rows;
	device->comMoveToScale(scaleX, scaleY); qDebug() << "(scaleX, scaleY) = (" << scaleX << ", " << scaleY << ")\n";
	device->comHitDown();

	Sleep(t * 1000);
	scaleX = (7.0 + toLoc.x) / pt.cols;
	scaleY = ((double)toLoc.y) / pt.rows;
	device->comMoveToScale(scaleX, scaleY); qDebug() << "(scaleX, scaleY) = (" << scaleX << ", " << scaleY << ")\n";
	//Sleep(3 * 1000);
	device->comHitUp();
	Sleep(t * 1000);

	device->comMoveToScale(0, 0); // 返回原点
	Sleep(t * 1000 / 3);
}

void usrGameController::click_at(cv::Point& loc, cv::Mat& pt) {

	scaleX = (7.0 + loc.x) / pt.cols;
	scaleY = ((double)loc.y) / pt.rows;
	if (scaleX > 1 || scaleY > 1 || scaleX < 0 || scaleY < 0) { qDebug() << "\nwarning pos out of bound!\n"; return; }
	device->comMoveToScale(scaleX, scaleY); qDebug() << "(scaleX, scaleY) = (" << scaleX << ", " << scaleY << ")\n";
	Sleep(1.5 * 1000);
	device->comHitDown();
	Sleep(100);
	device->comHitUp();
	Sleep(1.5 * 1000);
	device->comMoveToScale(0, 0); // 返回原点
	Sleep(1.5 * 1000);
}

/****************模板匹配***************/
void find_who_is_at_pos(Point& pos, int &seg_cnt) {

	Mat src, seg, result;
	double dSize = d;
	src = frame(Rect(pos - Point(d / 2, d / 2), Size(dSize, dSize)));

	// 全部子块都放进来匹配，比较谁的相似度最高
	double* minValues = new double[mode*mode];
	for (int seg_num = 0; seg_num < mode*mode; ++seg_num) {
		// 切出seg
		updateFilenames(seg_num);
		seg = imread(file_seg);
		cv::resize(seg, seg, Size(seg.cols / 2, seg.rows / 2));

		// 匹配src与seg
		_match(src, seg, result, minVal, maxVal, minLoc, maxLoc, matchLoc);

		// 保存minVal
		minValues[seg_num] = minVal;
	}

	// 找到最小的值
	double maxNum = *std::min_element(minValues, minValues + mode * mode);
	seg_cnt = std::find(minValues, minValues + mode * mode, maxNum) - minValues;
}

void find_accurate_pos_at_pos(Point& pos, int &seg_cnt, Point& rstLoc) {

	Mat src, seg, result;
	double dSize = 3 * d;
	Point tl = pos - Point(dSize / 2, dSize / 2);
	Point br = pos + Point(dSize / 2, dSize / 2);
	if (tl.x < 0) tl.x = 0;
	if (tl.y < 0) tl.y = 0;
	if (br.x > 540) br.x = 540;
	if (br.y > 960) br.y = 960;
	src = frame(Rect(tl, br));

	// 切出seg
	updateFilenames(seg_cnt);
	seg = imread(file_seg);
	cv::resize(seg, seg, Size(seg.cols / 2, seg.rows / 2));

	// 匹配src与seg
	_match(src, seg, result, minVal, maxVal, minLoc, maxLoc, matchLoc);
	
	rstLoc = tl + Point(matchLoc.x + seg.cols / 2, matchLoc.y + seg.rows / 2);
}

void find_where_is_seg_k(int seg_num, Point& rstLoc, int CHECK_TYPE) {
	// 切出src
	Mat src, seg, result, src_display;
	src = imread("frame.png");

	switch (CHECK_TYPE) {
	case CHECK_UP_MARGIN: src = src(Rect(0, 0, 540, Y0 / 2)); break;
	case CHECK_BOTTOM_MARGIN: src = src(Rect(0, Y1 / 2, 540, 960)); break;
	}
	cv::resize(src, src, Size(src.cols / alpha, src.rows / alpha));

	// 切出seg
	updateFilenames(seg_num);
	seg = imread(file_seg);
	cv::resize(seg, seg, Size(seg.cols / alpha / 2, seg.rows / alpha / 2));

	// 匹配src与seg
	_match(src, seg, result, minVal, maxVal, minLoc, maxLoc, matchLoc);

	// 保存匹配结果rstLoc
	rstLoc = matchLoc + Point(seg.cols / 2, seg.rows / 2);
	rstLoc *= alpha;
	if (CHECK_TYPE == CHECK_BOTTOM_MARGIN)
		rstLoc += Point(0, Y1 / 2);
}

void _match(Mat& src, Mat& seg, Mat& result, double& minVal, double& maxVal,
	Point& minLoc, Point& maxLoc, Point& matchLoc) {

	matchTemplate(src, seg, result, TM_SQDIFF);
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	matchLoc = minLoc;

}

/*****************对上下边缘的检查************************/
bool checkMargin(int CHECK_UP_OR_DOWN) {

	cout << "正在对上下边缘做检查" << endl;
	Mat A = imread("frame.png"), AA = imread(file_init);
	cv::resize(AA, AA, A.size());
	Mat C = cv::abs(A - AA);

	Mat MarginPic;
	if (CHECK_UP_OR_DOWN == CHECK_UP_MARGIN)
		MarginPic = C(Rect(Point(0, 0), Point(540, Y0 / 2)));
	else
		MarginPic = C(Rect(Point(0, Y1 / 2), Point(540, 960)));

	cvtColor(MarginPic, MarginPic, cv::COLOR_RGB2GRAY);
	threshold(MarginPic, MarginPic, 10, 255, THRESH_BINARY);
	erode(MarginPic, MarginPic, cv::getStructuringElement(MORPH_RECT, Size(5, 5)));

	Mat threshold_output = MarginPic;
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	if (contours.size() == 0) return false;

	// 获取 bounded Rect 及其 center 坐标
	Rect bddRect; bool hasMarginRect = false;
	vector< vector<Point> >  contours_poly(contours.size());
	for (int i = 0; i < contours.size(); ++i) {
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		bddRect = boundingRect(Mat(contours_poly[i]));

		if (bddRect.area() > 2000 && bddRect.width < 2*d) {

			rstLoc.x = (bddRect.tl().x + bddRect.br().x) / 2;
			rstLoc.y = (bddRect.tl().y + bddRect.br().y) / 2; 
			cout << "bddRect: " << bddRect << endl;
			cout << "rstLoc: " << rstLoc << endl;
			/*rectangle(MarginPic, bddRect, Scalar::all(255), 5, 8, 0);
			putText(MarginPic, num, rstLoc, 20, 2, Scalar(255, 0, 255), 2);
			imshow("MarginPic", MarginPic);
			waitKey(-1);*/
			//system("pause");

			hasMarginRect = true; break;
		
		}
	}
	
	if (CHECK_UP_OR_DOWN == CHECK_BOTTOM_MARGIN) // 注意如果是下边缘检查时y坐标要加上剪切常数
		rstLoc.y += Y1 / 2;

	return hasMarginRect;
}

/****************判断当前状态***************/
double check_match(int& cnt, int pic_num) {
	// 用于判断第cnt块区域与图片pic是否相似
	// pic_num = -1 时表示与default匹配；其他表示与子块seg[pic_num]匹配

	// 把cntLoc区域切出来作为src
	Mat src, seg, result;
	frame = imread("frame.png");

	double xi0, yi0;
	xi0 = X0 + d * (1 + 2.0 * (seg_cnt % mode));
	yi0 = Y0 + d * (1 + 2.0 * (seg_cnt / mode));

	double dSize = 0.7*d;
	src = frame(Rect(Point(xi0 / 2 - dSize / 2, yi0 / 2 - dSize / 2), Size(dSize, dSize)));

	// 切出seg
	updateFilenames(pic_num);
	if (pic_num == -1) {
		seg = imread(file_play);
		seg = seg(Rect(Point(xi0 - dSize / 2, yi0 - dSize / 2), Size(dSize, dSize)));
	}
	else
		seg = imread(file_seg);

	cv::resize(seg, seg, Size(seg.cols / 2, seg.rows / 2));

	// 匹配src与seg
	_match(src, seg, result, minVal, maxVal, minLoc, maxLoc, matchLoc);

	return minVal;
}

void usrGameController::checkFrameState(cv::Mat& _pt_global) {

	Mat  templ, result;
	frame = imread("frame.png");
	resize(frame, frame, Size(540 / alpha, 960 / alpha));

	templ = imread(game_stop_symbol);
	cv::resize(templ, templ, Size(templ.cols / 2 / alpha, templ.rows / 2 / alpha));
	_match(frame, templ, result, minVal, maxVal, minLoc, maxLoc, matchLoc);
	qDebug() << "GAME_IN minVal: " << minVal << endl;
	if (minVal < 2e6) {
		STATE = GAME_IN;
		qDebug() << "GAME_IN matched success" << endl; return;
	}

	templ = imread(game_select01);
	cv::resize(templ, templ, Size(540 / alpha - 1, 960 / alpha - 1));
	_match(frame, templ, result, minVal, maxVal, minLoc, maxLoc, matchLoc);
	qDebug() << "GAME_SELECT01 minVal: " << minVal << endl;
	if (minVal < 1e9) {
		STATE = GAME_SELECT01;
		qDebug() << "GAME_SELECT01 matched success" << endl;
		// Click the game icon on home screen
		Point clickLoc = Point(672 / 2, 200 / 2);
		click_at(clickLoc, _pt_global); return;
	}

	templ = imread(game_select04);
	cv::resize(templ, templ, Size(540 / alpha - 1, 960 / alpha - 1));
	_match(frame, templ, result, minVal, maxVal, minLoc, maxLoc, matchLoc);
	qDebug() << "GAME_SELECT04 minVal: " << minVal << endl;
	if (minVal < 1e9) {
		STATE = GAME_SELECT04;
		qDebug() << "GAME_SELECT04 matched success" << endl;
		// select MODE-slider
		Point clickLoc = Point((76 + 92 * (mode - 2)) / 2, 1366 / 2);
		click_at(clickLoc, _pt_global);
		// select PLAY
		clickLoc = Point(554 / 2, 1706 / 2);
		click_at(clickLoc, _pt_global); return;
	}

	templ = imread(game_stop);
	cv::resize(templ, templ, Size(540 / alpha - 1, 960 / alpha - 1));
	_match(frame, templ, result, minVal, maxVal, minLoc, maxLoc, matchLoc);
	qDebug() << "GAME_STOP minVal: " << minVal << endl;
	if (minVal < 1e9) {
		STATE = GAME_STOP;
		qDebug() << "GAME_STOP matched success" << endl;
		click_at(STOP / 2, _pt_global); return;
	}

	
}

bool checkSuccess() {

	Mat  templ, result;
	frame = imread("frame.png");
	resize(frame, frame, Size(540 / alpha, 960 / alpha));
	templ = imread(game_success01);
	cv::resize(templ, templ, Size(540 / alpha - 1, 960 / alpha - 1));
	_match(frame, templ, result, minVal, maxVal, minLoc, maxLoc, matchLoc);
	qDebug() << "GAME_SUCCESS01 minVal: " << minVal << endl;
	if (minVal < 4e8) {
		STATE = GAME_SUCCESS;
		qDebug() << "GAME_SUCCESS01 matched success" << endl;
		return true;
	}
	templ = imread(game_success02);
	cv::resize(templ, templ, Size(540 / alpha - 1, 960 / alpha - 1));
	_match(frame, templ, result, minVal, maxVal, minLoc, maxLoc, matchLoc);
	qDebug() << "GAME_SUCCESS02 minVal: " << minVal << endl;
	if (minVal < 4e8) {
		STATE = GAME_SUCCESS;
		qDebug() << "GAME_SUCCESS02 matched success" << endl;
		return true;
	}

	updateFilenames(0);
	Mat A = imread("frame.png");
	Mat B = imread(file_play);
	resize(B, B, Size(B.cols / 2, B.rows / 2)); 
	Mat C = abs(B - A);
	C = C(Rect(Point(X0 / 2, Y0 / 2), Point(X1 / 2, Y1 / 2)));
	cvtColor(C, C, cv::COLOR_RGB2GRAY);
	threshold(C, C, 10, 255, THRESH_BINARY);
	erode(C, C, cv::getStructuringElement(MORPH_RECT, Size(3, 3)));
	dilate(C, C, cv::getStructuringElement(MORPH_RECT, Size(5, 5)));
	erode(C, C, cv::getStructuringElement(MORPH_RECT, Size(7, 7)));
	remainedPics = mode * mode - round(sum(C)[0] / 256.0 / (C.rows*C.cols)  * mode * mode);

	return false;
}

void updateFilenames(int seg_num = 0)
{
	// 写这个函数主要是因为我的模板图片的路径总是被我搬来搬去，不想换一次路径改一次了
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