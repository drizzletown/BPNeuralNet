// BPNetTest_MultiplyRule.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>
#include "../疯狂实践系列十三：BP神经网络/NeuralNet.h"
#include "../疯狂实践系列十三：BP神经网络/Mapping.h"
#include <Eigen>
#include <iostream>
using namespace Eigen;
using namespace std;


int main()
{
	vector<int> layer_neuron_num = { 2,50,1 };
	//给定数据，如函数体字符序列->函数名字符序列
	//1. 训练神经网络
	NeuralNet nn;
	nn.initNet(layer_neuron_num);

	vector<Vector_xd> x;
	//准备训练数据和实际目标值
	const int num = 11;
	double b[num][3] = { {2,5,10}, {3,6,18},{12,2,24},{1,6,6},{9,2,18},
						{8,12,96}, {4,7,28},{7,9,63},{1,10,10},{15,8,120},
						{220,3,660} };
	double tg[num] = {};
	for (int i = 0; i < num; i++) { tg[i] = b[i][2]; }

	MappingToN1_1 mping;  //MappingTo0_1 mping;
	double amin = 0, amax = 0;
	vector<Vector_xd> target = mping.map(tg, num, amin, amax);

	for (int i = 0; i < num; i++) {
		Vector_xd x1(2); x1 << b[i][0], b[i][1]; x.push_back(x1);
	}

	nn.setLearningRate(0.05);
	nn.setTrainNum(num);
	nn.setActFunctionType(ActFunctionType::TANH);
	nn.setTrainingType(TrainingType::MiniBatched);
	nn.training(x, target);

	vector<Vector_xd> prd;
	nn.pred(x, prd);
	mping.deMap(prd, amin, amax);

	for (int i = 0; i < x.size(); i++) {
		cout << "输入数据：" << endl;
		for (int j = 0; j < x[i].size(); j++)
			cout << x[i](j) << " ";
		cout << endl;

		cout << "目标值：" << endl;
		cout << tg[i] << " ";
		cout << endl;

		cout << "预测值：" << endl;
		cout << prd[i](0) << " ";
		cout << endl;

	}
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门提示: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
