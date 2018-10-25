#pragma once
#include <Eigen>
using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic> Matrix_xd;
typedef Matrix<double, Dynamic, 1> Vector_xd;

enum ActFunctionType { sigmod, ReLU, tanh, softplus, identity };
class ActFunction {
public:
	virtual Vector_xd f(Vector_xd vec) = 0;
	virtual Vector_xd df() = 0;
	virtual ~ActFunction() {}
};

class IdentityFunciton : public ActFunction {
public:
	virtual Vector_xd f(Vector_xd vec) {
		ivec = vec;
		return ivec;
	}
	virtual Vector_xd df() {
		Vector_xd divec = ivec;
		divec.setOnes();
		return divec;
	}
	Vector_xd ivec;
};


class SigmodFunciton : public ActFunction {
public:
	virtual Vector_xd f(Vector_xd vec) {
		sigvec = vec;
		sigvec = -sigvec;
		sigvec = sigvec.array().exp();
		sigvec = sigvec.array() + 1.0;
		sigvec = 1.0 / sigvec.array();
		return sigvec;
	}
	virtual Vector_xd df() {
		assert(sigvec.size > 0);
		return sigvec.array() *(1 - sigvec.array());
	}
private:
	Vector_xd sigvec;
};


class TanhFunciton : public ActFunction {
public:
	//return 2*sigmod(vec*2).array()-1;
	virtual Vector_xd f(Vector_xd vec) {
		SigmodFunciton sf;
		tanhvec = 2 * sf.f(vec * 2).array() - 1;
		return tanhvec;
	}
	virtual Vector_xd df() {
		assert(tanhvec.size > 0);
		return 1 - tanhvec.array()*tanhvec.array();
	}
private:
	Vector_xd tanhvec;
};


class ReLUFunciton : public ActFunction {
public:
	virtual Vector_xd f(Vector_xd vec) {
		reluvec = vec;
		for (int i = 0; i < reluvec.size(); i++)
			if (reluvec(i) < 0) reluvec(i) = 0;
		return reluvec;
	}
	virtual Vector_xd df() {
		assert(reluvec.size > 0);
		Vector_xd dreluvec = reluvec;
		for (int i = 0; i < dreluvec.size(); i++)
			if (dreluvec(i) < 0) dreluvec(i) = 0;
			else dreluvec(i) = 1;		
		return dreluvec;
	}
private:
	Vector_xd reluvec;
};


class SoftplusFunciton : public ActFunction {
public:
	virtual Vector_xd f(Vector_xd vec) {
		spvec = vec;
		spvec = spvec.array().exp() + 1;
		spvec = spvec.log();
		return spvec;
	}
	virtual Vector_xd df() {
		assert(spvec.size > 0);
		SigmodFunciton sf;		
		return sf.f();
	}
private:
	Vector_xd spvec;
};
