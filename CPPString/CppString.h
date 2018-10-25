#pragma once
#include <string>
#include <sstream>
#include <functional>
#include <algorithm>
#include <numeric>
#include <vector>
#include <iterator>
using namespace std;

class CppString : public string {
	

public:
	CppString(string cst) : string(cst) {}

	string toUpper() {
		string newstring(*this);
		transform(this->begin(), this->end(), newstring.begin(), toupper);
		return newstring;
	}

	string toLower() {
		string newstring(*this);
		transform(this->begin(), this->end(), newstring.begin(), tolower);
		return newstring;
	}

	string trim() {
		return trim(' ');
	}

	string trim(char ch) {
		string newstring(*this);
		newstring.erase(0, newstring.find_first_not_of(ch));
		newstring.erase(newstring.find_last_not_of(ch));
		return newstring;
	}

	CppString trim(CppString ch) {
		CppString newstring(*this);
		newstring.erase(0, newstring.find_first_not_of(ch));
		newstring.erase(newstring.find_last_not_of(ch));
		return newstring;
	}

	CppString del(char cs) {
		CppString newstring(*this);
		newstring.erase(remove_if(newstring.begin(), newstring.end(), bind2nd(equal_to<char>(), cs)), newstring.end());
		return newstring;
	}
	CppString del() {
		return del(' ').del('\t');
	}
	
	CppString replaceWith(CppString it, CppString with) {
		CppString newstring(*this);
		newstring.replace(newstring.find(it), it.size(), with);
		return newstring;
	}

	bool startWith(CppString cs) {
		return this->compare(0, cs.size(), cs) == 0;
	}

	bool endWith(CppString cs) {
		return this->compare(this->size()-cs.size(), cs.size(), cs) == 0;
	}

	int toInt() { return stoi(*this); }
	double toDouble() { return stod(*this); }
	double toBoolean() {
		bool b; istringstream(*this) >> boolalpha >> b;
		return b;
	}

	vector<CppString> split() {
		vector<CppString> vecs;
		istringstream iss(*this);
		vecs.assign(istream_iterator<string>(iss), istream_iterator<string>());
		return vecs;
	}

	static CppString join(vector<CppString> vecs) {
		return accumulate(vecs.begin(), vecs.end(), CppString(""));
	}

	static CppString join(vector<CppString> vecs, CppString cs) {
		return accumulate(vecs.begin(), vecs.end(), CppString(""));
	}

};
