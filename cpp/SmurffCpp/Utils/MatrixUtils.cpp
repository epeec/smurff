#include "MatrixUtils.h"

#include <numeric>
#include <set>
#include <vector>
#include <iterator>

#include <Utils/Error.h>

namespace smurff {

Matrix matrix_utils::dense_to_eigen(const TensorConfig& matrixConfig)
{
   THROWERROR_ASSERT_MSG(matrixConfig.isDense(), "matrix config should be dense");
   THROWERROR_ASSERT_MSG(matrixConfig.getNModes() == 2, "Invalid number of dimensions. Tensor can not be converted to matrix.");

   std::vector<float_type> float_values(matrixConfig.getValues().begin(), matrixConfig.getValues().end());
   return Eigen::Map<const Matrix>(float_values.data(), matrixConfig.getNRow(), matrixConfig.getNCol());
}

Matrix matrix_utils::dense_to_eigen(const DenseTensor& matrixAsTensor)
{
   THROWERROR_ASSERT_MSG(matrixAsTensor.getNModes() == 2, "Invalid number of dimensions. Tensor can not be converted to matrix.");

   std::vector<float_type> float_values(matrixAsTensor.getValues().begin(), matrixAsTensor.getValues().end());
   return Eigen::Map<const Matrix>(matrixAsTensor.getValues().data(), matrixAsTensor.getNRow(), matrixAsTensor.getNCol());
}

std::shared_ptr<MatrixConfig> matrix_utils::eigen_to_dense(const Matrix &eigenMatrix, NoiseConfig n)
{
   std::vector<double> values(eigenMatrix.data(),  eigenMatrix.data() + eigenMatrix.size());
   return std::make_shared<MatrixConfig>(eigenMatrix.rows(), eigenMatrix.cols(), values, n);
}

SparseMatrix matrix_utils::sparse_to_eigen(const TensorConfig& tensorConfig)
{
   THROWERROR_ASSERT_MSG(!tensorConfig.isDense(), "tensor config should be sparse");
   THROWERROR_ASSERT_MSG(tensorConfig.getNModes() == 2, "Invalid number of dimensions. Tensor can not be converted to matrix.");

   SparseMatrix out(tensorConfig.getNRow(), tensorConfig.getNCol());

   sparse_vec_iterator begin(tensorConfig, 0);
   sparse_vec_iterator end(tensorConfig, tensorConfig.getNNZ());

   out.setFromTriplets(begin, end);

   THROWERROR_ASSERT_MSG(out.nonZeros() == (int)tensorConfig.getNNZ(), "probable presence of duplicate records in " + tensorConfig.getFilename());

   return out;
}

SparseMatrix matrix_utils::sparse_to_eigen(const SparseTensor& matrixAsTensor)
{
   THROWERROR_ASSERT_MSG(matrixAsTensor.getNModes() == 2, "Invalid number of dimensions. Tensor can not be converted to matrix.");

   SparseMatrix out(matrixAsTensor.getNRow(), matrixAsTensor.getNCol());

   sparse_vec_iterator begin(matrixAsTensor, 0);
   sparse_vec_iterator end(matrixAsTensor, matrixAsTensor.getNNZ());

   out.setFromTriplets(begin, end);

   return out;
}

std::shared_ptr<MatrixConfig> matrix_utils::eigen_to_sparse(const SparseMatrix &X, NoiseConfig n, bool isScarce)
{
   std::uint64_t nrow = X.rows();
   std::uint64_t ncol = X.cols();
   std::uint64_t nnz = X.nonZeros();

   auto ret = std::make_shared<MatrixConfig>(false, false, isScarce, nrow, ncol, nnz, n);

   for (int k = 0; k < X.outerSize(); ++k)
      for (SparseMatrix::InnerIterator it(X,k); it; ++it)
      {
         ret->getRows().push_back(it.row());
         ret->getCols().push_back(it.col());
         ret->getValues().push_back(it.value());
      }

   return ret;
}

std::ostream& matrix_utils::operator << (std::ostream& os, const MatrixConfig& mc)
{
   const std::vector<std::uint32_t>& rows = mc.getRows();
   const std::vector<std::uint32_t>& cols = mc.getCols();
   const std::vector<double>& values = mc.getValues();

   if(rows.size() != cols.size() || rows.size() != values.size())
   {
      THROWERROR("Invalid sizes");
   }

   os << "rows: " << std::endl;
   for(std::uint64_t i = 0; i < rows.size(); i++)
      os << rows[i] << ", ";
   os << std::endl;

   os << "cols: " << std::endl;
   for(std::uint64_t i = 0; i < cols.size(); i++)
      os << cols[i] << ", ";
   os << std::endl;

/*
   os << "columns: " << std::endl;
   for(std::uint64_t i = 0; i < columns.size(); i++)
   {
      for(std::uint64_t j = 0; j<columns[i].size(); j++)
         os << columns[i][j] << ", ";
      os << std::endl;
   }
   os << std::endl;
*/
   os << "values: " << std::endl;
   for(std::uint64_t i = 0; i < values.size(); i++)
      os << values[i] << ", ";
   os << std::endl;

   os << "NRow: " << mc.getNRow() << " NCol: " << mc.getNCol() << std::endl;

   SparseMatrix X(mc.getNRow(), mc.getNCol());

   std::vector<Eigen::Triplet<double> > triplets;
   for(std::uint64_t i = 0; i < mc.getNNZ(); i++)
      triplets.push_back(Eigen::Triplet<double>(rows[i], cols[i], values[i]));

   os << "NTriplets: " << triplets.size() << std::endl;

   X.setFromTriplets(triplets.begin(), triplets.end());

   os << X << std::endl;

   return os;
}

bool matrix_utils::equals(const Matrix& m1, const Matrix& m2, double precision)
{
   if (m1.rows() != m2.rows() || m1.cols() != m2.cols())
      return false;

   for (Eigen::Index i = 0; i < m1.rows(); i++)
   {
      for (Eigen::Index j = 0; j < m1.cols(); j++)
      {
         Matrix::Scalar m1_v = m1(i, j);
         Matrix::Scalar m2_v = m2(i, j);

         if (std::abs(m1_v - m2_v) > precision)
            return false;
      }
   }

   return true;
}

bool matrix_utils::equals_vector(const Vector& v1, const Vector& v2, double precision)
{
   if (v1.size() != v2.size())
      return false;

   for (auto i = 0; i < v1.size(); i++)
   {
      if (std::abs(v1(i) - v2(i)) > precision)
         return false;
   }

   return true;
}

} // end namespace