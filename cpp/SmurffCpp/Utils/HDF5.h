#pragma once

#include <string>
#include <memory>
#include <vector>
#include <set>
#include <cstdint>

#include <highfive/H5File.hpp>

#include <SmurffCpp/Types.h>

namespace h5 = HighFive;

namespace smurff {
   class HDF5 
   {
   protected:
      mutable h5::File m_file;
      mutable h5::Group m_group;

   public:
      HDF5(h5::File file, h5::Group group) : m_file(file), m_group(group) {}
      HDF5(h5::File file) : m_file(file) {}

      bool hasDataSet(const std::string &section, const std::string& tag) const;

      std::shared_ptr<Matrix> getMatrix(const std::string &section, const std::string& tag) const;
      std::shared_ptr<Vector> getVector(const std::string &section, const std::string& tag) const;
      std::shared_ptr<SparseMatrix> getSparseMatrix(const std::string &section, const std::string& tag) const;

      void putMatrix(const std::string &section, const std::string& tag, const Matrix &) const;
      void putSparseMatrix(const std::string &section, const std::string& tag, const SparseMatrix &) const;
   };
}
