// BPNetTest_Iris.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include "../疯狂实践系列十三：BP神经网络/NeuralNet.h"
#include "../疯狂实践系列十三：BP神经网络/Mapping.h"
#include <Eigen>
#include <iostream>
#include <fstream>
#include <filesystem>
using namespace Eigen;
using namespace std;


void getIris(vector<Vector_xd> & dataset, vector<double> & target, string fileName);

int main()
{
	vector<Vector_xd> dataset;
	vector<double> target;

	getIris(dataset, target, "./iris/iris.data");

	vector<int> layer_neuron_num = { 4,10,1 };
	NeuralNet nn;
	nn.initNet(layer_neuron_num);

	MappingToN1_1 mping;  //MappingTo0_1 mping;
	double amin = 0, amax = 0;
	vector<Vector_xd> mtarget = mping.map(target, amin, amax);

	nn.setTrainNum(dataset.size());
	nn.setLearningRate(0.001);
	nn.setActFunctionType(ActFunctionType::TANH);
	nn.setTrainingType(TrainingType::MiniBatched);
	nn.setEpoch(5000);
	nn.training(dataset, mtarget);

	vector<Vector_xd> prd;
	nn.pred(dataset, prd);
	mping.deMap(prd, amin, amax);

	for (int i = 0; i < dataset.size(); i++) {
		/*cout << "输入数据：" << endl;
		for (int j = 0; j < dataset[i].size(); j++)
			cout << dataset[i](j) << " ";
		cout << endl;*/
		cout << "目标值-预测值：" << target[i] << "\t" << prd[i](0)  <<endl;

	}
}


void getIris(vector<Vector_xd> & dataset, vector<double> & target, string fileName) {
	ifstream ifs(fileName);
	while (!ifs.eof()) {
		Vector_xd vec(4);
		string sdval;
		int i = 0;
		for (; i < 4; i++) {
			getline(ifs, sdval, ',');
			if (!isdigit(sdval[0])) break;
			double dval = stof(sdval); //atof(sdval);			
			vec(i) = dval;
		}
		if (i < 4) break;
		dataset.push_back(vec);
		getline(ifs, sdval, '\n');
		int label = stoi(sdval);
		target.push_back(label);
	}
	ifs.close();
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
