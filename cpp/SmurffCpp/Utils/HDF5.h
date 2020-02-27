#pragma once

#include <string>
#include <memory>
#include <vector>
#include <set>
#include <cstdint>

#include <highfive/H5File.hpp>

#include <SmurffCpp/Utils/ConfigFile.h>

namespace h5 = HighFive;

namespace smurff {
   class HDF5 : public ConfigFile
   {
   protected:
      h5::Group m_group;

   private:
      h5::Group getGroup(const std::string &) const;
      h5::Group addGroup(const std::string &);

      template <typename T>
      T getInternal(const std::string &section, const std::string& tag, const T &default_value) const
      {
         T value;

         if (this->getGroup(section).hasAttribute(tag))
            this->getGroup(section).getAttribute(tag).read(value);
         else 
            value = default_value;

         return value;
      }

      template <typename T>
      void putInternal(const std::string &section, const std::string& tag, const T &value)
      {
         this->addGroup(section).createAttribute(tag, value);
      }

   public:
      HDF5(h5::Group group) : m_group(group) {}

      virtual ~HDF5() {}

      bool hasDataSet(const std::string &section, const std::string& tag) const override;
      bool hasSection(const std::string &section) const override;

      int         get(const std::string &section, const std::string& tag, const int         &default_value) const override
      { return getInternal(section, tag, default_value); }

      size_t      get(const std::string &section, const std::string& tag, const size_t      &default_value) const override
      { return getInternal(section, tag, default_value); }

      double      get(const std::string &section, const std::string& tag, const double      &default_value) const override
      { return getInternal(section, tag, default_value); }

      bool        get(const std::string &section, const std::string& tag, const bool        &default_value) const override
      { return getInternal(section, tag, default_value); }

      std::string get(const std::string &section, const std::string& tag, const std::string &default_value) const override
      { return getInternal(section, tag, default_value); }

      void read(const std::string &section, const std::string& tag, Vector &) const override;
      void read(const std::string &section, const std::string& tag, Matrix &) const override;
      void read(const std::string &section, const std::string& tag, SparseMatrix &) const override;
      void read(const std::string &section, const std::string& tag, DenseTensor &) const override;
      void read(const std::string &section, const std::string& tag, SparseTensor &) const override;

      void put(const std::string &section, const std::string& tag, const int         &value) override
      { putInternal(section, tag, value); }

      void put(const std::string &section, const std::string& tag, const size_t      &value) override
      { putInternal(section, tag, value); }

      void put(const std::string &section, const std::string& tag, const double      &value) override
      { putInternal(section, tag, value); }

      void put(const std::string &section, const std::string& tag, const bool        &value) override
      { putInternal(section, tag, value); }

      void put(const std::string &section, const std::string& tag, const std::string &value) override
      { putInternal(section, tag, value); }

      void write(const std::string &section, const std::string& tag, const Vector &) override;
      void write(const std::string &section, const std::string& tag, const Matrix &) override;
      void write(const std::string &section, const std::string& tag, const SparseMatrix &) override;
      void write(const std::string &section, const std::string& tag, const DenseTensor &) override;
      void write(const std::string &section, const std::string& tag, const SparseTensor &) override;
   };
}
