// test6.12_yoloV3.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include<opencv.hpp>
#include <dnn.hpp>

using namespace cv;
using namespace std;
using namespace dnn;

std::vector<std::string> classes;

float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;        // Width of network's input image
int inpHeight = 416;       // Height of network's input image

std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net)
{
	static std::vector<cv::String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		std::vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		std::vector<cv::String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

void postprocess(cv::Mat& frame, std::vector<cv::Mat>& outs);
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);

int main()
{

	VideoCapture cap("..//YOLOV3_VIDEO.mp4");

	if (!cap.isOpened())return -1;


	//coco数据集的名称文件，80类
	string classesFile = "..//coco.names";
	//yolov3网络模型文件
	String yolov3_model = "..//yolov3.cfg";
	//权重
	String weights = "..//yolov3.weights";

	//将coco.names中的80类名称转换为vector形式
	std::ifstream classNamesFile(classesFile.c_str());
	if (classNamesFile.is_open())
	{
		std::string className = "";
		// getline (istream&  is, string& str)
		//is为输入，从is中读取读取的字符串保存在string类型的str中，如果没有读入字符返回false，循环结束
		while (std::getline(classNamesFile, className)) {
			classes.push_back(className);
		}
	}
	else {
		std::cout << "can not open classNamesFile" << std::endl;
	}

	cv::dnn::Net net = cv::dnn::readNetFromDarknet(yolov3_model, weights);

	net.setPreferableBackend(DNN_BACKEND_DEFAULT);
	net.setPreferableTarget(DNN_TARGET_CPU);

	cv::Mat frame;

	while (1)
	{
		cap >> frame;

		if (frame.empty()) {
			std::cout << "frame is empty!!!" << std::endl;
			return -1;
		}

		//创建yolo输入数据
		cv::Mat blob;
		cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0, 0, 0), true, false);

		//输入网络
		net.setInput(blob);

		//定义输出结果保存容器
		std::vector<cv::Mat> outs;
		//前向传输获得结果
		net.forward(outs, getOutputsNames(net));

		//后处理，非极大值抑制，绘制检测框
		postprocess(frame, outs);

		cv::imshow("frame", frame);

		if (cv::waitKey(10) == 27)
		{
			break;
		}
	}

	return 0;

}

void postprocess(cv::Mat& frame, std::vector<cv::Mat>& outs)
{
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		//
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			cv::Point classIdPoint;
			double confidence;

			//获得得分最高的结果的分值和位置
			cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(cv::Rect(left, top, width, height));
			}
		}
	}

	//非极大值抑制，去除置信度较小的检测结果
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
	//绘制检测框
	cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255));

	//获得识别结果的类名称，以及置信度
	std::string label = cv::format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}
	else
	{
		std::cout << "classes is empty..." << std::endl;
	}

	//绘制标签
	int baseLine;
	cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseLine);
	top = std::max(top, labelSize.height);
	cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}
// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
