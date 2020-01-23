#include "MatrixUtils.h"

#include <numeric>
#include <set>
#include <vector>
#include <iterator>

#include <Utils/Error.h>

namespace smurff {

Matrix matrix_utils::dense_to_eigen(const MatrixConfig& matrixConfig)
{
   if(!matrixConfig.isDense())
   {
      THROWERROR("matrix config should be dense");
   }

   std::vector<float_type> float_values(matrixConfig.getValues().begin(), matrixConfig.getValues().end());
   return Eigen::Map<const Matrix>(float_values.data(), matrixConfig.getNRow(), matrixConfig.getNCol());
}

std::shared_ptr<MatrixConfig> matrix_utils::eigen_to_dense(const Matrix &eigenMatrix, NoiseConfig n)
{
   std::vector<double> values(eigenMatrix.data(),  eigenMatrix.data() + eigenMatrix.size());
   return std::make_shared<MatrixConfig>(eigenMatrix.rows(), eigenMatrix.cols(), values, n);
}

struct sparse_vec_iterator
{
  sparse_vec_iterator(const MatrixConfig& matrixConfig, int pos)
     : config(matrixConfig), pos(pos) {}

  const MatrixConfig& config;
  int pos;

  bool operator!=(const sparse_vec_iterator &other) const {
     THROWERROR_ASSERT(&config == &other.config);
     return pos != other.pos;
  }

  sparse_vec_iterator &operator++() { pos++; return *this; }

  typedef Eigen::Triplet<float_type> T;
  T v;

  T* operator->() {
     // also convert from 1-base to 0-base
     uint32_t row = config.getRows()[pos];
     uint32_t col = config.getCols()[pos];
     float_type val = config.getValues()[pos];
     v = T(row, col, val);
     return &v;
  }
};

SparseMatrix matrix_utils::sparse_to_eigen(const MatrixConfig& matrixConfig)
{
   if(matrixConfig.isDense())
   {
      THROWERROR("matrix config should be sparse");
   }

   SparseMatrix out(matrixConfig.getNRow(), matrixConfig.getNCol());

   sparse_vec_iterator begin(matrixConfig, 0);
   sparse_vec_iterator end(matrixConfig, matrixConfig.getNNZ());

   out.setFromTriplets(begin, end);

   THROWERROR_ASSERT_MSG(out.nonZeros() == (int)matrixConfig.getNNZ(), "probable presence of duplicate records in " + matrixConfig.getFilename());

   return out;
}

std::shared_ptr<MatrixConfig> matrix_utils::eigen_to_sparse(const SparseMatrix &X, NoiseConfig n, bool isScarce)
{
   std::uint64_t nrow = X.rows();
   std::uint64_t ncol = X.cols();

   std::vector<uint32_t> rows;
   std::vector<uint32_t> cols;
   std::vector<double> values;

   for (int k = 0; k < X.outerSize(); ++k)
   {
      for (SparseMatrix::InnerIterator it(X,k); it; ++it)
      {
         rows.push_back(it.row());
         cols.push_back(it.col());
         values.push_back(it.value());
      }
   }

   return std::make_shared<MatrixConfig>(nrow, ncol, rows, cols, values, n, isScarce);
}

std::ostream& matrix_utils::operator << (std::ostream& os, const MatrixConfig& mc)
{
   const std::vector<std::uint32_t>& rows = mc.getRows();
   const std::vector<std::uint32_t>& cols = mc.getCols();
   const std::vector<double>& values = mc.getValues();
   const std::vector<std::uint32_t>& columns = mc.getColumns();

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

   os << "columns: " << std::endl;
   for(std::uint64_t i = 0; i < columns.size(); i++)
      os << columns[i] << ", ";
   os << std::endl;

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