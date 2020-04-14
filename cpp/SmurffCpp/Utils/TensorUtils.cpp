#include <numeric>

#include "TensorUtils.h"
#include "MatrixUtils.h"

#include <Utils/Error.h>

namespace smurff {

std::ostream& tensor_utils::operator << (std::ostream& os, const TensorConfig& tc)
{
   const std::vector<double>& values = tc.getValues();

   os << "columns: " << std::endl;
   for (int j = 0; j < tc.getNModes(); ++j)
   {
      const std::vector<std::uint32_t> &column = tc.getColumn(j);
      for (std::uint64_t i = 0; i < column.size(); i++)
         os << column[i] << ", ";
      os << std::endl;
   }

   os << "values: " << std::endl;
   for(std::uint64_t i = 0; i < values.size(); i++)
      os << values[i] << ", ";
   os << std::endl;

   if(tc.getNModes() == 2)
   {
      os << "dims: " << tc.getDims()[0] << " " << tc.getDims()[1] << std::endl;

      SparseMatrix X = matrix_utils::sparse_to_eigen(tc);
      os << X << std::endl;
   }

   return os;
}

Matrix tensor_utils::slice( const TensorConfig& tensorConfig
                           , const std::array<std::uint64_t, 2>& fixedDims
                           , const std::unordered_map<std::uint64_t, std::uint32_t>& dimCoords)
{
   if (fixedDims[0] == fixedDims[1])
   {
      THROWERROR("fixedDims should contain 2 unique dimension numbers");
   }

   for (const std::uint64_t& fd : fixedDims)
   {
      if (fd > tensorConfig.getNModes() - 1)
      {
         THROWERROR("fixedDims should contain only valid for tensorConfig dimension numbers");
      }
   }

   if (dimCoords.size() != (tensorConfig.getNModes() -  2))
   {
      THROWERROR("dimsCoords.size() should be the same as tensorConfig.getNModes() - 2");
   }

   for (const std::unordered_map<std::uint64_t, std::uint32_t>::value_type& dc : dimCoords)
   {
      if (dc.first == fixedDims[0] || dc.first == fixedDims[1])
      {
         THROWERROR("dimCoords and fixedDims should not intersect");
      }

      if (dc.first >= tensorConfig.getNModes())
      {
         THROWERROR("dimCoords should contain only valid for tensorConfig dimension numbers");
      }

      if (dc.second >= tensorConfig.getDims()[dc.first])
      {
         THROWERROR("dimCoords should contain valid coord values for corresponding dimensions");
      }
   }

   Matrix sliceMatrix(tensorConfig.getDims()[fixedDims[0]], tensorConfig.getDims()[fixedDims[1]]);
   for (std::size_t i = 0; i < tensorConfig.getValues().size(); i++)
   {
      bool dimCoordsMatchColumns =
         std::accumulate( dimCoords.begin()
                        , dimCoords.end()
                        , true
                        , [&](bool acc, const std::unordered_map<std::uint64_t, std::uint32_t>::value_type& dc)
                          {
                             return acc & (tensorConfig.getColumn(dc.first)[i] == dc.second);
                          }
                        );

      if (dimCoordsMatchColumns)
      {
         std::uint32_t d0_coord = tensorConfig.getColumn(fixedDims[0])[i];
         std::uint32_t d1_coord = tensorConfig.getColumn(fixedDims[1])[i];
         sliceMatrix(d0_coord, d1_coord) = tensorConfig.getValues()[i];
      }
   }
   return sliceMatrix;
}

} // end namespace smurff
