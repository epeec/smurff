#include "MatrixConfig.h"

#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/IO/IDataWriter.h>
#include <SmurffCpp/DataMatrices/IDataCreator.h>
#include <Utils/Error.h>

namespace smurff {

MatrixConfig::MatrixConfig(bool isDense, bool isBinary, bool isScarce,
                std::uint64_t nrow, std::uint64_t ncol, std::uint64_t nnz,
                const NoiseConfig& noiseConfig, PVec<> pos)
   : TensorConfig(isDense, isBinary, isScarce, 2, nnz, noiseConfig, pos)
{
   m_dims = {nrow, ncol};
}

// Dense double matrix constructos
MatrixConfig::MatrixConfig( std::uint64_t nrow
                          , std::uint64_t ncol
                          , const double* values
                          , const NoiseConfig& noiseConfig
                          , PVec<> pos
                          )
   : TensorConfig({nrow, ncol}, values, noiseConfig, pos)
{
   setData(matrix_utils::dense_to_eigen(*this));
}

// Sparse double matrix constructor
MatrixConfig::MatrixConfig( std::uint64_t nrow
                          , std::uint64_t ncol
                          , std::uint64_t nnz
                          , const std::uint32_t* rows
                          , const std::uint32_t* cols
                          , const double* values
                          , const NoiseConfig& noiseConfig
                          , bool isScarce
                          , PVec<> pos
                          )
   : TensorConfig({nrow, ncol}, nnz, {rows, cols}, values, noiseConfig, isScarce, pos)
{
   setData(matrix_utils::sparse_to_eigen(*this));
}

// Sparse binary matrix constructors
MatrixConfig::MatrixConfig( std::uint64_t nrow
                          , std::uint64_t ncol
                          , std::uint64_t nnz
                          , const std::uint32_t* rows
                          , const std::uint32_t* cols
                          , const NoiseConfig& noiseConfig
                          , bool isScarce
                          , PVec<> pos
                          )
   : TensorConfig({nrow, ncol}, nnz, {rows, cols}, noiseConfig, isScarce, pos)
{
   setData(matrix_utils::sparse_to_eigen(*this));
}

//
// other methods
//

std::shared_ptr<Data> MatrixConfig::create(std::shared_ptr<IDataCreator> creator) const
{
   //have to use dynamic cast here but only because shared_from_this() can only return base pointer even from child
   return creator->create(std::dynamic_pointer_cast<const MatrixConfig>(shared_from_this()));
}

void MatrixConfig::write(std::shared_ptr<IDataWriter> writer) const
{
   //have to use dynamic cast here but only because shared_from_this() can only return base pointer even from child
   writer->write(std::dynamic_pointer_cast<const MatrixConfig>(shared_from_this()));
}
} // end namespace smurff
