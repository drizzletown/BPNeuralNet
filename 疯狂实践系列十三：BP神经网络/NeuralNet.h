#pragma once

#include <random>
#include <cmath>
#include <vector>
#include <Eigen>
#include <iostream>
using namespace Eigen;
using namespace std;

typedef Matrix<double,Dynamic, Dynamic> Matrix_xd;
typedef Matrix<double, Dynamic, 1> Vector_xd;

typedef std::tr1::ranlux64_base_01 MyEng; 
typedef std::normal_distribution<double> MyNormDist; 
typedef std::uniform_real_distribution<double> MyUnifDist; 

enum ActFunctionType { SIGMOD, RELU, TANH, IDENTITY };
namespace ActFunction {
	Vector_xd sigmod(const Vector_xd& vec) {
		Vector_xd sigvec = vec;
		sigvec = -sigvec;
		sigvec = sigvec.array().exp();
		sigvec = sigvec.array() + 1.0;
		sigvec = 1.0 / sigvec.array();
		return sigvec;
	}
	Vector_xd dsigmod(const Vector_xd& fx) {
		return fx.array()*(1.0 - fx.array());
	}

	Vector_xd tanh(const Vector_xd& vec) {
		Vector_xd tanhvec = vec.array() * 2;
		tanhvec = 2 * sigmod(tanhvec).array() - 1;
		return tanhvec;
	}
	Vector_xd dtanh(const Vector_xd& fx) {
		return (1.0 - fx.array()*fx.array());
	}

	Vector_xd ReLU(const Vector_xd& vec) {
		Vector_xd reluvec = vec;
		for (int i = 0; i < reluvec.size(); i++)
			if (reluvec(i) < 0) reluvec(i) = 0;
		return reluvec;
	}
	Vector_xd dReLU(const Vector_xd& fx) {
		Vector_xd dreluvec = fx;
		for (int i = 0; i < dreluvec.size(); i++)
			if (dreluvec(i) < 0) dreluvec(i) = 0;
			else dreluvec(i) = 1;
		return dreluvec;
	}

	Vector_xd identity(const Vector_xd& vec) {
		Vector_xd reluvec = vec;
		return reluvec;
	}
	Vector_xd didentity(const Vector_xd& fx) {
		Vector_xd divec = fx;
		divec.setOnes();
		return divec;
	}
};


typedef Vector_xd(*PtrActFunction)(const Vector_xd& vec);

class NeuralNet
{
	
public:
	NeuralNet() { 
		epoch = 10000;  
		learning_rate = 0.005; 
		loss_threshold = 0.001;
		trainingNum = -1;
	}

	void pred(const vector<Vector_xd>& x, vector<Vector_xd>& prd) {
		for (int j = 0; j < x.size(); j++) {
			setInputLayer(x[j]);
			goAhead();
			Vector_xd opt = getOutput();
			prd.push_back(opt);
		}
	}

	//全批量训练
	void batchedTraining(const vector<Vector_xd>& x, const vector<Vector_xd>& target) {
	}
	//随机批量训练
	void miniBatchedTraining(const vector<Vector_xd>& x, const vector<Vector_xd>& target) {
	}
	//在线训练
	void stochasticTraining(const vector<Vector_xd>& x, const vector<Vector_xd>& target) {
		double oError = 0.0;
		for(int i=0; i<epoch; i++) {
			double tError = 0.0;
			for(int j =0; j<x.size(); j++) {
				setInputLayer(x[j]);
				goAhead();
				double loss = calcDelta(target[j]);
				tError += loss;
				if (loss < loss_threshold) continue;
				goBack();			
			}
			cout << "训练遍数：" <<i << endl;
			cout << "总体误差：" << tError << endl;
			if (tError < loss_threshold) break;
			if (fabs(tError - oError) < 1e-10) break;
			oError = tError;
		}
	}

	

	void goAhead() {
		for(int i=0; i<neuronNums.size()-1; i++) {
			int layer = i + 1;
			PtrActFunction paf = getActFunction();
			Vector_xd vec = getWeight(layer).transpose()*getLayer(i) + getBias(layer);
			layers[layer] = paf(vec);
		}
	}


	void goBack() {
		updateWeight();
	}

	double calcDelta(const Vector_xd& tk) {
		//计算输出层delta
		Vector_xd& ok = getOutput();
		double loss = (tk - ok).dot(tk - ok)/(2*tk.size());
		//delta[delta.size()-1] = ok.array()*(1.0-ok.array())*(tk-ok).array();
		delta[delta.size() - 1] = getDActFunction()(ok).array()*(tk - ok).array();
		
		//计算隐层delta
		for(int i=layers.size()-2; i>=1; i--) {
			Vector_xd& oh = getLayer(i);
			//Vector_xd ohdev = oh.array()*(1.0-oh.array());
			Vector_xd ohdev = getDActFunction()(oh);
			Vector_xd sumdelta = getWeight(i+1)*getDelta(i+1);
			getDelta(i) = ohdev.array()*sumdelta.array();
		}
		return loss;
	}	

	void updateWeight() {
		//Delta{wji} = mu * Delta_j * x_ji
		//wji = wji + Delta{wji}
		for (int i = 1; i < layers.size(); i++) {
			getWeight(i) += learning_rate * getLayer(i-1) * getDelta(i).transpose();
			getBias(i) += learning_rate * getDelta(i);
		}
	}

#pragma region 激活函数
	void setActFunctionType(ActFunctionType aft) {
		actFuncType = aft;
	}

	PtrActFunction getActFunction() {
		switch (actFuncType)
		{
		case ActFunctionType::SIGMOD:
			return  ActFunction::sigmod;
			break;
		case ActFunctionType::RELU:
			return ActFunction::ReLU;
			break;
		case ActFunctionType::TANH:
			return ActFunction::tanh;
			break;
		case ActFunctionType::IDENTITY:
			return ActFunction::identity;
			break;
		default:
			return ActFunction::sigmod;
			break;
		}
	}
	PtrActFunction getDActFunction() {
		switch (actFuncType)
		{
		case ActFunctionType::SIGMOD:
			return ActFunction::dsigmod;
			break;
		case ActFunctionType::RELU:
			return ActFunction::dReLU;
			break;
		case ActFunctionType::TANH:
			return ActFunction::dtanh;
			break;
		case ActFunctionType::IDENTITY:
			return ActFunction::didentity;
			break;
		default:
			return ActFunction::dsigmod;
			break;
		}
	}

	Vector_xd sigmod(Vector_xd& vec) {
		Vector_xd sigvec = vec;
		sigvec = -sigvec;
		sigvec = sigvec.array().exp();
		sigvec = sigvec.array() + 1.0;
		sigvec = 1.0 / sigvec.array();
		return sigvec;
	}
	Vector_xd dsigmod(Vector_xd& fx) {
		return fx.array()*(1.0 - fx.array());
	}

	Vector_xd tanh(Vector_xd& vec) {
		Vector_xd tanhvec = vec.array() * 2;
		tanhvec = 2 * sigmod(tanhvec).array() - 1;
		return tanhvec;
	}
	Vector_xd dtanh(Vector_xd& fx) {
		return (1.0 - fx.array()*fx.array());
	}

	Vector_xd ReLU(Vector_xd& vec) {
		Vector_xd reluvec = vec;
		for (int i = 0; i < reluvec.size(); i++)
			if (reluvec(i) < 0) reluvec(i) = 0;
		return reluvec;
	}
	Vector_xd dReLU(Vector_xd& fx) {
		Vector_xd dreluvec = fx;
		for (int i = 0; i < dreluvec.size(); i++)
			if (dreluvec(i) < 0) dreluvec(i) = 0;
			else dreluvec(i) = 1;
		return dreluvec;
	}

	Vector_xd identity(Vector_xd& vec) {
		Vector_xd reluvec = vec;
		return reluvec;
	}
	Vector_xd didentity(Vector_xd& fx) {
		Vector_xd divec = fx;
		divec.setOnes();
		return divec;
	}

#pragma endregion



#pragma region 打印输出
	void printvec(Vector_xd v) const {
		for (int i = 0; i < v.size(); i++)
			cout << v(i) << " ";
		cout << endl;
	}

	void printmtx(Matrix_xd m) const {
		cout << m << endl;
	}

	void printDelta() const {
		cout << "Delta:" << endl;
		for (int i = 0; i < delta.size(); i++) {
			for (int j = 0; j < delta[i].size(); j++) {
				cout << delta[i](j) << " ";
			}
			cout << endl;
		}
		cout << endl;
	}

	void printLayers() {
		cout << "网络层:" << endl;
		for (int i = 0; i < layers.size(); i++) {
			for (int j = 0; j < layers[i].size(); j++) {
				cout << layers[i](j) << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
	void printBias() const {
		cout << "偏置:" << endl;
		for (int i = 0; i < b.size(); i++) {
			for (int j = 0; j < b[i].size(); j++) {
				cout << b[i](j) << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
	void printWeight() const {
		cout << "权值:" << endl;
		for (int i = 0; i < w.size(); i++) {
			cout << "第" << i << "层：" << w[i] << endl;
		}
		cout << endl;
	}
#pragma endregion
	   	 
#pragma region 初始化

	void initNet(vector<int> neuronNums) {
		this->neuronNums = neuronNums;
		int layer1Num, layer2Num;
		for (int i = 0; i < neuronNums.size() - 1; i++) {
			layer1Num = neuronNums[i];
			layer2Num = neuronNums[i + 1];

			Vector_xd layer(layer1Num);
			layers.push_back(layer);

			Vector_xd layerDelta(layer2Num);
			delta.push_back(layerDelta);

			Vector_xd layer2_b(layer2Num);
			b.push_back(layer2_b);

			Matrix_xd layer1to2_w(layer1Num, layer2Num);
			w.push_back(layer1to2_w);
		}
		Vector_xd layer(layer2Num);
		layers.push_back(layer);


		initBias();
		initWeights();
	}



	void initBias() {
		for (int i = 0; i < b.size(); i++) {
			b[i].setZero();
		}
	}

	void initWeights() {
		int n = trainingNum;
		if (n < 0) n = 10;
		//MyNormDist dist(0.0, 1.0/n);  
		MyUnifDist dist(-0.1, 1.0/n);
		MyEng engine;

		for (int i = 0; i < w.size(); i++) {
			Matrix_xd& wi = w[i];
			for (int j = 0; j < wi.rows(); j++) {
				for (int k = 0; k < wi.cols(); k++) {
					wi(j, k) = dist(engine);
				}
			}
		}
	}
#pragma endregion
		
#pragma region Getters & Setters
	void setTrainNum(int n) { trainingNum = n; }

	void setInputLayer(Vector_xd x) {
		layers[0] = x;
	}

	void setEpoch(int ep) { epoch = ep; }
	void setLearningRate(double rt) { learning_rate = rt; }
	void setLossThreshold(double th) { loss_threshold = th; }
	


	Vector_xd& getOutput() {
		return layers.back();
	}

	Matrix_xd& getWeight(int layer) {
		assert(layer >= 1);
		return w[layer - 1];
	}
	Vector_xd& getDelta(int layer) {
		assert(layer >= 1);
		return delta[layer - 1];
	}
	Vector_xd& getBias(int layer) {
		assert(layer >= 1);
		return b[layer - 1];
	}

	Vector_xd& getLayer(int layer) {
		assert(layer >= 0);
		return layers[layer];
	}
#pragma endregion
	
	

private:
	int trainingNum;
	ActFunctionType actFuncType;
	int epoch;
	double loss_threshold;
	double learning_rate; 
	vector<Vector_xd> delta;	//与偏置的结构相同，除输入层外每个神经元一个delta值

private:
	vector<int> neuronNums;		//每层的神经元个数
	vector<Vector_xd> layers;   //每层神经元的输出值
	vector<Vector_xd> b;		//每层的偏置向量，第0层没有偏置
	vector<Matrix_xd> w;		//每层的权值矩阵，第0层没有权重，w[k]表示第k层的权值矩阵(rows, cols)，rows是k-1层的神经元个数，cols是k层的神经元个数
};

