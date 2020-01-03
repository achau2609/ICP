// Eigen.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <Eigen/Dense>
#include <opencv2\opencv.hpp>
#include <vector>

using Eigen::Vector3d;
using Eigen::RowVector3d;

void solve(std::vector<Vector3d>& p, std::vector<Vector3d>& p_) {

	int count = p.size();
	// 1. centroid 1 & 2
	Vector3d pc; pc << 0, 0, 0;
	Vector3d pc_; pc_ << 0, 0, 0;

	for (int i = 0; i < count; i++) {
		pc << pc + p[i];
	}
	for (int i = 0; i < count; i++) {
		pc_ << pc_ + p[i];
	}

	// divide for centroid
	pc = pc / count;
	pc_ = pc_ / count;

	// subtract centroid from two sets of points

	std::vector<Vector3d> q;
	std::vector<Vector3d> q_;
	for (int i = 0; i < count; i++) {
		q.push_back(p[i] - pc);
	}
	for (int i = 0; i < count; i++) {
		q_.push_back(p_[i] - pc_);
	}

	// 2. define W = sum(q * q_ transpose)
	// q_ transpose
	std::vector<RowVector3d> q_t;
	for (int i = 0; i < q_.size(); i++) {
		q_t.push_back(q_[i]);
	}

	// calculate W iteratively
	Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
	Eigen::Matrix3d W_ = Eigen::Matrix3d::Zero();;
	for (int i = 0; i < count; i++) {
		// temp matrices
		Vector3d colmat = q[i];
		RowVector3d rowmat = q_t[i];
		//fill temp W
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				W_(j, k) = colmat[j] * rowmat[k];
			}
		}
		W << W + W_;
	}


	// 3. run svd on W
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);

	// U,Sigma and V transpose
	Eigen::Matrix3d U = svd.matrixU();
	Eigen::Matrix3d V = svd.matrixV();
	Eigen::Vector3d Sigma = svd.singularValues();

	// Value of R
	Eigen::Matrix3d R;
	R << U * V.transpose();

	// 4. find t
	Eigen::MatrixXd t;
	t = pc - (R * pc_);

	Eigen::Matrix3d sigmaRecons = Eigen::Matrix3d::Zero();
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (j == i) {
				sigmaRecons(i, j) = Sigma[i];
			}
		}
	}

	std::cout << "This is W:" << std::endl << W << std::endl << std::endl;
	std::cout << "After doing SVD on the matrix, we obtained U, V and sigma." << std::endl << std::endl;
	std::cout << "U is:" << std::endl << U << std::endl << std::endl;
	std::cout << "V is:" << std::endl << V << std::endl << std::endl;
	std::cout << "Sigma is:" << std::endl << sigmaRecons << std::endl << std::endl;

	std::cout << "We then obtain R and t." << std::endl << std::endl;
	std::cout << "R:" << std::endl << R << std::endl << std::endl;
	std::cout << "t:" << std::endl << t << std::endl << std::endl;

}

std::vector<Vector3d> genpts() {
	std::vector<Vector3d> set;
	int x, y, z;
	Vector3d point;
	for (int i = 0; i < 1000; i++) {
		x = rand() % 20;
		y = rand() % 20;
		z = rand() % 20;
		point << x, y, z;
		set.push_back(point);
	}
	return set;
}

int main()
{
	/*
	std::vector<Vector3d> p;
	Vector3d pt1; pt1 << 0, 1, 1;
	p.push_back(pt1);
	Vector3d pt2; pt2 << 0, -1, 0;
	p.push_back(pt2);
	Vector3d pt3; pt3 << 1, 0, 1;
	p.push_back(pt3);
	Vector3d pt4; pt4 << 0, 0, 1;
	p.push_back(pt4);

	std::vector<Vector3d> p_;
	pt1 << 1, 0, 0;
	p_.push_back(pt1);
	pt2 << -1, 1, 0;
	p_.push_back(pt2);
	pt3 << 0, 0, 0;
	p_.push_back(pt3);
	pt4 << 1, -1, 1;
	p_.push_back(pt4);

	solve(p, p_); */

	std::vector<Vector3d> r = genpts();
	std::vector<Vector3d> r_ = genpts();

	solve(r, r_);
}
