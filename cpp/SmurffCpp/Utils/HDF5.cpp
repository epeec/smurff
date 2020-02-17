#include <iostream>

#include <SmurffCpp/Utils/HDF5.h>

#include <Utils/Error.h>
#include <Utils/StringUtils.h>


namespace smurff {

bool HDF5::hasDataSet(const std::string& section, const std::string& tag) const
{
   if (!m_group.exist(section)) return false;
   auto section_group = m_group.getGroup(section);
   return (section_group.exist(tag));
}

std::shared_ptr<Matrix> HDF5::getMatrix(const std::string& section, const std::string& tag) const
{
   auto dataset = m_group.getGroup(section).getDataSet(tag);
   std::vector<size_t> dims = dataset.getDimensions();
   Matrix data(dims[0], dims[1]);
   dataset.read(data.data());

   if (data.IsRowMajor)
      return std::make_shared<Matrix>(data);

   // convert to ColMajor if needed (HDF5 always stores row-major)
   return std::make_shared<Matrix>(Eigen::Map<Eigen::Matrix<
            typename Matrix::Scalar,
            Matrix::RowsAtCompileTime,
            Matrix::ColsAtCompileTime,
            Matrix::ColsAtCompileTime==1?Eigen::ColMajor:Eigen::RowMajor,
            Matrix::MaxRowsAtCompileTime,
            Matrix::MaxColsAtCompileTime>>(data.data(), dims[0], dims[1]));
   
}

std::shared_ptr<Vector> HDF5::getVector(const std::string& section, const std::string& tag) const
{

   auto dataset = m_group.getGroup(section).getDataSet(tag);
   std::vector<size_t> dims = dataset.getDimensions();
   THROWERROR_ASSERT(dims[1] == 1);
   Vector data(dims[0]);
   dataset.read(data.data());
   return std::make_shared<Vector>(data);
}

std::shared_ptr<SparseMatrix> HDF5::getSparseMatrix(const std::string& section, const std::string& tag) const
{
   auto sparse_group = m_group.getGroup(section).getGroup(tag); 

   std::string format;
   sparse_group.getAttribute("h5sparse_format").read(format);
   THROWERROR_ASSERT(( SparseMatrix::IsRowMajor && format == "csr") || \
                     (!SparseMatrix::IsRowMajor && format == "csc"));
   
   std::vector<Eigen::Index> shape(2);
   sparse_group.getAttribute("h5sparse_shape").read(shape);
   SparseMatrix X(shape.at(0), shape.at(1));
   X.makeCompressed();

   auto data = sparse_group.getDataSet("data");
   THROWERROR_ASSERT(data.getDataType() == h5::AtomicType<SparseMatrix::value_type>());
   X.resizeNonZeros(data.getElementCount());
   data.read(X.valuePtr());

   auto indptr = sparse_group.getDataSet("indptr");
   THROWERROR_ASSERT(indptr.getDataType() == h5::AtomicType<SparseMatrix::Index>());
   indptr.read(X.outerIndexPtr());

   auto indices = sparse_group.getDataSet("indices");
   THROWERROR_ASSERT(indices.getDataType() == h5::AtomicType<SparseMatrix::Index>());
   indices.read(X.innerIndexPtr());

   return std::make_shared<SparseMatrix>(X);
}

void HDF5::putMatrix(const std::string& section, const std::string& tag, const Matrix &M) const
{
   if (!m_group.exist(section))
      m_group.createGroup(section);

   h5::Group group = m_group.getGroup(section);
   h5::DataSet dataset = group.createDataSet<Matrix::Scalar>(tag, h5::DataSpace::From(M));

   Eigen::Ref<
        const Eigen::Matrix<
            Matrix::Scalar,
            Matrix::RowsAtCompileTime,
            Matrix::ColsAtCompileTime,
            Matrix::ColsAtCompileTime==1?Eigen::ColMajor:Eigen::RowMajor,
            Matrix::MaxRowsAtCompileTime,
            Matrix::MaxColsAtCompileTime>,
        0,
        Eigen::InnerStride<1>> row_major(M);

    dataset.write(row_major.data());
}

void HDF5::putSparseMatrix(const std::string& section, const std::string& tag, const SparseMatrix &X) const
{
   if (!m_group.exist(section))
      m_group.createGroup(section);

   h5::Group sparse_group = m_group.getGroup(section).createGroup(tag);

   sparse_group.createAttribute<std::string>("h5sparse_format", (SparseMatrix::IsRowMajor ? "csr" : "csc"));
   std::vector<Eigen::Index> shape{X.innerSize(), X.outerSize()};
   sparse_group.createAttribute<Eigen::Index>("h5sparse_shape", h5::DataSpace::From(shape)).write(shape);

   auto data = sparse_group.createDataSet<SparseMatrix::value_type>("data", h5::DataSpace(X.nonZeros()));
   data.write(X.valuePtr());

   auto indptr = sparse_group.createDataSet<SparseMatrix::Index>("indptr", h5::DataSpace(X.outerSize() + 1));
   indptr.write(X.outerIndexPtr());

   auto indices = sparse_group.createDataSet<SparseMatrix::Index>("indices", h5::DataSpace(X.nonZeros()));
   indices.write(X.innerIndexPtr());
}

} // end namespace smurff
