#include "MatrixUtils.h"

#include <numeric>
#include <set>
#include <vector>
#include <iterator>

#include <Utils/Error.h>

namespace smurff {

Matrix matrix_utils::dense_to_eigen(const DenseTensor& matrixAsTensor)
{
   THROWERROR_ASSERT_MSG(matrixAsTensor.getNModes() == 2, "Invalid number of dimensions. Tensor can not be converted to matrix.");

   std::vector<float_type> float_values(matrixAsTensor.getValues().begin(), matrixAsTensor.getValues().end());
   return Eigen::Map<const Matrix>(float_values.data(), matrixAsTensor.getNRow(), matrixAsTensor.getNCol());
}

Matrix matrix_utils::make_dense(
          const std::vector<std::uint64_t> &dims,
          const std::vector<double> &values
)
{
   return dense_to_eigen(smurff::DenseTensor(dims, values));
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

SparseMatrix matrix_utils::make_sparse(
    const std::vector<std::uint64_t> &dims,
    const std::vector<std::vector<std::uint32_t>> &columns,
    const std::vector<double> &values)
{
   return sparse_to_eigen(smurff::SparseTensor(dims, columns, values));
}

bool matrix_utils::equals(const Matrix& m1, const Matrix& m2, double epsilon)
{
   if (m1.rows() != m2.rows() || m1.cols() != m2.cols())
      return false;

   for (Eigen::Index i = 0; i < m1.rows(); i++)
   {
      for (Eigen::Index j = 0; j < m1.cols(); j++)
      {
         auto abs_m1 = std::abs(m1(i, j));
         auto abs_m2 = std::abs(m2(i, j));
         auto abs_diff = std::abs(m2(i, j) - m1(i, j));

			if ((abs_diff / (abs_m1 + abs_m2)) > epsilon)
         {
            std::cout << "@" << i << "," << j << ":"
                      << abs_m1 << " != " << abs_m2 << "(diff: " << abs_diff << ")" 
                      << std::endl;
            return false;
         }
      }
   }

   return true;
}

// make sure they are both row-vectors
bool matrix_utils::equals_vector(const Vector& v1, const Vector& v2, double epsilon)
{
   return equals(v1, v2, epsilon);
}


} // end namespace