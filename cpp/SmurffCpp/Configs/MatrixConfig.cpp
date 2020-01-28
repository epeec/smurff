#include "MatrixConfig.h"

#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/IO/IDataWriter.h>
#include <SmurffCpp/DataMatrices/IDataCreator.h>
#include <Utils/Error.h>

namespace smurff {

MatrixConfig::MatrixConfig(bool isDense, bool isBinary, bool isScarce,
                std::uint64_t nrow, std::uint64_t ncol, std::uint64_t nnz,
                const NoiseConfig& noiseConfig)
   : TensorConfig(isDense, isBinary, isScarce, 2, nnz, noiseConfig)
{
   m_dims = {nrow, ncol};
}

// Dense double matrix constructos
MatrixConfig::MatrixConfig( std::uint64_t nrow
                          , std::uint64_t ncol
                          , const double* values
                          , const NoiseConfig& noiseConfig
                          )
   : TensorConfig({nrow, ncol}, values, noiseConfig)
{}

// Sparse double matrix constructor
MatrixConfig::MatrixConfig( std::uint64_t nrow
                          , std::uint64_t ncol
                          , std::uint64_t nnz
                          , const std::uint32_t* rows
                          , const std::uint32_t* cols
                          , const double* values
                          , const NoiseConfig& noiseConfig
                          , bool isScarce
                          )
   : TensorConfig({nrow, ncol}, nnz, {rows, cols}, values, noiseConfig, isScarce)
{}

// Sparse binary matrix constructors
MatrixConfig::MatrixConfig( std::uint64_t nrow
                          , std::uint64_t ncol
                          , std::uint64_t nnz
                          , const std::uint32_t* rows
                          , const std::uint32_t* cols
                          , const NoiseConfig& noiseConfig
                          , bool isScarce
                          )
   : TensorConfig({nrow, ncol}, nnz, {rows, cols}, noiseConfig, isScarce)
{}

//
// other methods
//

std::uint64_t MatrixConfig::getNRow() const
{
   return m_dims.operator[](0);
}

std::uint64_t MatrixConfig::getNCol() const
{
   return m_dims.operator[](1);
}

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
