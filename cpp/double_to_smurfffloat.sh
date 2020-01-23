#!/bin/sh

FILES=$(find . -name '*.cpp' -o -name '*.hpp' -o -name '*.h')

for F in $FILES
do
	gsed -i -e "
	s/Eigen::MatrixXd/Matrix/g;
	s/Eigen::VectorXd/Vector/g;
	s/Eigen::ArrayXXd/Array/g;
	s/Eigen::SparseMatrix<double>/SparseMatrix/g;
	" $F

	gsed -i -e "
	s@include <Eigen/Core>@include <SmurffCpp/Types.h>@;
	s@include <Eigen/Dense>@include <SmurffCpp/Types.h>@;
	s@include <Eigen/SparseCore>@include <SmurffCpp/Types.h>@;
	s@include <Eigen/Sparse>@include <SmurffCpp/Types.h>@;
	" $F
done

git checkout SmurffCpp/Types.h

