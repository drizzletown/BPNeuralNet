// ���ʵ��ϵ��ʮ����BP������.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include "NeuralNet.h"
#include "Mapping.h"
#include <Eigen>
#include <iostream>
using namespace Eigen;
using namespace std;

void printvec(Vector_xd v);
void printmtx(Matrix_xd m);

int _tmain(int argc, _TCHAR* argv[])
{
	vector<int> layer_neuron_num = {2,10,1};
	//�������ݣ��纯�����ַ�����->�������ַ�����
	//1. ѵ��������
	NeuralNet nn;
	nn.initNet(layer_neuron_num);
	nn.setActFunctionType(ActFunctionType::TANH);
	//nn.setActFunctionType(ActFunctionType::SIGMOD);
	
	
	vector<Vector_xd> x;
	//׼��ѵ�����ݺ�ʵ��Ŀ��ֵ
	const int num = 11;
	double b[num][3] = { {2,5,10}, {3,6,18},{12,2,24},{1,6,6},{9,2,18},
						{8,12,96}, {4,7,28},{7,9,63},{1,10,10},{15,8,120},
						{220,3,660} };
	double tg[num] = {};
	for (int i = 0; i < num; i++) { tg[i] = b[i][2]; }

	MappingToN1_1 mping;  //MappingTo0_1 mping;
	double amin=0, amax=0;
	vector<Vector_xd> target = mping.map(tg, num, amin, amax);
	
	for (int i = 0; i < num; i++) {
		Vector_xd x1(2); x1 << b[i][0],b[i][1]; x.push_back(x1);
	}
	
	nn.setTrainNum(num);
	nn.stochasticTraining(x, target);

	vector<Vector_xd> prd;
	nn.pred(x, prd);
	mping.deMap(prd, amin, amax);

	for (int i = 0; i < x.size(); i++) {
		cout << "�������ݣ�" << endl;
		for (int j = 0; j < x[i].size(); j++)
			cout << x[i](j) << " ";
		cout << endl;

		cout << "Ŀ��ֵ��" <<  endl;
		cout << tg[i] << " ";
		cout << endl;

		cout << "Ԥ��ֵ��" << endl;
		cout << prd[i](0) << " ";
		cout << endl;

	}


	//nn.training(/*ѵ������*/);
	//nn.save();
	//2. ������������һЩԤ�⹤��
	//nn.pred(/*��������*/);



	return 0;
}


void printvec(Vector_xd v) {
	for (int i = 0; i < v.size(); i++)
		cout << v(i) << " ";
	cout << endl;
}

void printmtx(Matrix_xd m) {
	cout << m << endl;
}