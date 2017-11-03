int usrGameController::usrProcessImage(cv::Mat& img)
{
	cv::Size imgSize(img.cols, img.rows - UP_CUT);
	if (imgSize.height <= 0 || imgSize.width <= 0)
	{
		qDebug() << "Invalid image. Size:" << imgSize.width << "x" << imgSize.height;
		return -1;
	}

	//截取图像边缘
	cv::Mat pt = img(cv::Rect(0, 0, imgSize.width, imgSize.height));
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
	int mode = 2;// 8*8=64模式

	// 视频流存放在该路径下
	imwrite("frame.png", img);
	frame = imread("frame.png");
	/*string fname = "E:/frames/frame_";
	fname += ('0' + mode*mode / 10); fname += ('0' + mode*mode % 10); fname += '_';
	fname += ('0' + counter / 100); fname += ('0' + counter / 10 % 10); fname += ('0' + counter % 10);
	fname += (".png");
	imwrite(fname, img);*/

	//记住, img 就是当前帧的数据
	checkMatchedState();

	const int x0 = 52, y0 = 320, x1 = 500, y1 = 768;// 目标正方形区域
	double d = (x1 - x0) / 2 / mode;

	bool* finishedPic = new bool[mode*mode];
	for (int i = 0; i < mode*mode; ++i) finishedPic[i] = false;
	char input;
	bool isSuccess = checkSuccess(); 

	if (STATE == GAME_IN && !isSuccess) {
		// 进入游戏运行状态
		match_template(mode);
		success_game_in_cnt++;
		out << "successfully matched " << success_game_in_cnt << " time" << endl;
		qDebug() << "successfully matched " << success_game_in_cnt << " time" << endl;
		for (int i = 0; i < mode*mode; ++i) {
			if (finishedPic[i] == false && matchedPics[i].similarity < 1e-5) {
				out << "successfully comHitDown" << endl;
				double scaleX, scaleY;
				scaleX = ((double)matchedPics[i].pt.x  * alpha) / pt.cols;
				scaleY = ((double)matchedPics[i].pt.y * alpha) / pt.rows;
				out << "scaleX:" << scaleX << "scaleY" << scaleY << endl;
				qDebug() << "scaleX:" << scaleX << "scaleY" << scaleY << endl;
				device->comMoveToScale(scaleX, scaleY);
				device->comHitDown();

				scaleX = ((double)x0 + d * (1 + 2 * (matchedPics[i].index % mode))) / pt.cols;
				scaleY = ((double)y0 + d * (1 + 2 * (matchedPics[i].index / mode))) / pt.rows;
				out << "scaleX:" << scaleX << "scaleY" << scaleY << endl;
				qDebug() << "\nscaleX:" << scaleX << "scaleY" << scaleY << endl;
				device->comMoveToScale(scaleX, scaleY);
				device->comHitUp();

				finishedPic[i] = true;
			}
			qDebug() << "successfully moved one pic" << endl;
			device->comMoveToScale(0, 0);//return original pos
			system("pause");
		}
	}

	if (isSuccess) {
		qDebug() << " \n\n\nAll successfully matched, you can have fun debugging now!" << endl;
		device->comMoveToScale(0, 0);//return original pos
		system("pause");
	}

	return 0;
}

/****************模板匹配***************/
void match_template(int mode) {
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
		cout << "第" << i << "个拼图的参数\nminValue=" << minVal << "\tminLoc=(" << minLoc.x << ", " << minLoc.y << ")\n";
		out << "第" << i << "个拼图的参数\nminValue=" << minVal << "\tminLoc=(" << minLoc.x << ", " << minLoc.y << ")\n";

		resultPoints[i] = Point(matchLoc.x + templ.cols / 2, matchLoc.y + templ.rows / 2);
		rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(255), 2, 8, 0);
		putText(img_display, num, Point(matchLoc.x, matchLoc.y + templ.rows / 1.5), 6, 1, Scalar(255, 0, 255), 2);
	}

	imshow("Source Image", img);
	imwrite(file_final, img_display);
	namedWindow(file_final);
	imshow(file_final, img_display);

	//namedWindow("template", WINDOW_AUTOSIZE);
	//imshow("template", templ);

	// 排序结果
	matchedPics = new resultpt[mode*mode];
	double val; int pos;
	for (int i = 0; i < mode*mode; ++i) {
		val = similarityArr[i]; pos = 0;
		for (int j = 0; j < mode*mode; ++j)
			if (j != i && val < similarityArr[j]) pos++;
		if (matchedPics[pos].similarity == val) pos++;
		matchedPics[pos] = resultpt(resultPoints[i], i, val);
	}
	out << "matchedPics:\n";
	for (int i = 0; i < mode*mode; ++i) {
		out << i << ": " << matchedPics[i].similarity << endl;
	}
	out << "similarityArr:\n";
	for (int i = 0; i < mode*mode; ++i) {
		out << i << ": " << similarityArr[i] << endl;
	}
	//本函数可提供一个全局接口 resultPoints[mode*mode] 即最终所有拼图子块的中心
	//本函数可提供一个全局接口 matchedPics[mode*mode] 即依相似度排序后的拼图子块中心
	//结果不完全正确，可以先取相似度最高的子图先移动，再逐步调整
}

/****************判断当前状态***************/
void checkMatchedState() {
	Mat  templ, result;
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
	if (minVal < 1e-8) STATE = GAME_DOOR, qDebug() << "GAME_DOOR matched success" << endl, out << "GAME_DOOR matched success" << endl;

	templ = imread(file_init);
	matchTemplate(frame, templ, result, TM_SQDIFF);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	minVal = fabs(minVal);
	qDebug() << "GAME_INIT minVal: " << minVal << endl;
	out << "GAME_INIT minVal: " << minVal << endl;
	if (minVal < 1e-7) STATE = GAME_INIT, qDebug() << "GAME_INIT matched success" << endl, out << "GAME_INIT matched success" << endl;

	Mat in_square(frame, Rect(Point(60, 320), Point(500, 760)));
	Mat in_square_templ = imread("E:/in_square.png");
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
		imwrite(file_play, frame);
	}
}

bool checkSuccess() {
	Mat  templ, result;
	int mode;
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;

	string file_init("E:/init02_02.png");

	templ = imread(file_init);
	matchTemplate(frame, templ, result, TM_SQDIFF);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	minVal = fabs(minVal);

	if (minVal < 1e-7) {
		qDebug() << "ALL matched success" << endl;
		out << "ALL matched success" << endl;
		return true;
	}
	else
		return false;

}