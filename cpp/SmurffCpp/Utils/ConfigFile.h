#pragma once

#include <string>

#include <SmurffCpp/Types.h>

namespace smurff {
   class ConfigFile 
   {
   public:
      virtual ~ConfigFile() {}

      virtual bool hasDataSet(const std::string &section, const std::string& tag) const = 0;
      virtual bool hasSection(const std::string &section) const = 0;

      virtual int         get(const std::string &section, const std::string& tag, const int         &default_value) const = 0;
      virtual size_t      get(const std::string &section, const std::string& tag, const size_t      &default_value) const = 0;
      virtual double      get(const std::string &section, const std::string& tag, const double      &default_value) const = 0;
      virtual bool        get(const std::string &section, const std::string& tag, const bool        &default_value) const = 0;
      virtual std::string get(const std::string &section, const std::string& tag, const std::string &default_value) const = 0;

      virtual void read(const std::string &section, const std::string& tag, Vector &) const = 0;
      virtual void read(const std::string &section, const std::string& tag, Matrix &) const = 0;
      virtual void read(const std::string &section, const std::string& tag, SparseMatrix &) const = 0;
      virtual void read(const std::string &section, const std::string& tag, DenseTensor &) const = 0;
      virtual void read(const std::string &section, const std::string& tag, SparseTensor &) const = 0;

      virtual void put(const std::string &section, const std::string& tag, const int         &value) = 0;
      virtual void put(const std::string &section, const std::string& tag, const size_t      &value) = 0;
      virtual void put(const std::string &section, const std::string& tag, const double      &value) = 0;
      virtual void put(const std::string &section, const std::string& tag, const bool        &value) = 0;
      virtual void put(const std::string &section, const std::string& tag, const std::string &value) = 0;

      virtual void write(const std::string &section, const std::string& tag, const Vector &) = 0;
      virtual void write(const std::string &section, const std::string& tag, const Matrix &) = 0;
      virtual void write(const std::string &section, const std::string& tag, const SparseMatrix &) = 0;
      virtual void write(const std::string &section, const std::string& tag, const DenseTensor &) = 0;
      virtual void write(const std::string &section, const std::string& tag, const SparseTensor &) = 0;
   };
}
