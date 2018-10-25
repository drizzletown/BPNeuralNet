#pragma once
#include <vector>
#include <Eigen>
using namespace Eigen;
using namespace std;

typedef Matrix<double, Dynamic, 1> Vector_xd;

class Mapping {
public:
	virtual vector<Vector_xd> map(double a[], int n, double &amin, double &amax) = 0;
	virtual void deMap(vector<Vector_xd>& a, double amin, double amax) = 0;
};

class MappingTo0_1 : public Mapping {
public:
	vector<Vector_xd> map(double a[], int n, double &amin, double &amax) {
		amin = a[0], amax = a[0];
		for (int i = 0; i < n; i++) {
			if (a[i] > amax) amax = a[i];
			if (a[i] < amin) amin = a[i];
		}
		vector<Vector_xd> ret;
		for (int i = 0; i < n; i++) {
			Vector_xd v(1); v << (a[i] - amin) / (amax - amin);
			ret.push_back(v);
		}
		return ret;
	}


	void deMap(vector<Vector_xd>& a, double amin, double amax) {
		for (int i = 0; i < a.size(); i++) {
			a[i] = a[i].array()*(amax - amin) + amin;
		}
	}
};

class MappingToN1_1 : public Mapping {
public:
	vector<Vector_xd> map(double a[], int n, double &amin, double &amax) {
		amin = a[0], amax = a[0];
		for (int i = 0; i < n; i++) {
			if (a[i] > amax) amax = a[i];
			if (a[i] < amin) amin = a[i];
		}
		vector<Vector_xd> ret;
		for (int i = 0; i < n; i++) {
			Vector_xd v(1); v << 1.8*(a[i] - amin) / (amax - amin)-0.9;
			ret.push_back(v);
		}
		return ret;
	}


	void deMap(vector<Vector_xd>& a, double amin, double amax) {

		for (int i = 0; i < a.size(); i++) {
			a[i] = ((a[i].array()+0.9)*(amax - amin))/1.8+amin;
		}
	}
};