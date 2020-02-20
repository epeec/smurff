#pragma once

#include <string>
#include <boost/property_tree/ptree.hpp>

namespace pt = boost::property_tree;

#include <SmurffCpp/Utils/ConfigFile.h>

namespace smurff 
{
class INIFile : public ConfigFile
{
public:
   ~INIFile() {}

private:
   pt::ptree m_tree;
   pt::ptree makeUnique(const pt::ptree &pt);

public:
   void read(const std::string &filename);
   void write(const std::string &filename);

private:
   template <typename T>
   T getInternal(const std::string &section, const std::string &name, const T &default_value) const
   {
      return m_tree.get<T>(section + "." + name, default_value);
   }

   template <typename T>
   void putInternal(const std::string &section, const std::string &tag, const T &value)
   {
      m_tree.put(section + "." + tag, value);
   }

public:
   bool hasDataSet(const std::string &section, const std::string &tag) const override;
   bool hasSection(const std::string &section) const override;

   int get(const std::string &section, const std::string &tag, const int &default_value) const override
   {
      return getInternal(section, tag, default_value);
   }

   size_t get(const std::string &section, const std::string &tag, const size_t &default_value) const override
   {
      return getInternal(section, tag, default_value);
   }

   double get(const std::string &section, const std::string &tag, const double &default_value) const override
   {
      return getInternal(section, tag, default_value);
   }

   bool get(const std::string &section, const std::string &tag, const bool &default_value) const override
   {
      return getInternal(section, tag, default_value);
   }

   std::string get(const std::string &section, const std::string &tag, const std::string &default_value) const override
   {
      return getInternal(section, tag, default_value);
   }

   void read(const std::string &section, const std::string &tag, Vector &) const override;
   void read(const std::string &section, const std::string &tag, Matrix &) const override;
   void read(const std::string &section, const std::string &tag, SparseMatrix &) const override;
   void read(const std::string &section, const std::string &tag, SparseTensor &) const override;

   void put(const std::string &section, const std::string &tag, const int &value) override
   {
      putInternal(section, tag, value);
   }

   void put(const std::string &section, const std::string &tag, const size_t &value) override
   {
      putInternal(section, tag, value);
   }

   void put(const std::string &section, const std::string &tag, const double &value) override
   {
      putInternal(section, tag, value);
   }

   void put(const std::string &section, const std::string &tag, const bool &value) override
   {
      putInternal(section, tag, value);
   }

   void put(const std::string &section, const std::string &tag, const std::string &value) override
   {
      putInternal(section, tag, value);
   }

   void write(const std::string &section, const std::string &tag, const Vector &) override;
   void write(const std::string &section, const std::string &tag, const Matrix &) override;
   void write(const std::string &section, const std::string &tag, const SparseMatrix &) override;
   void write(const std::string &section, const std::string &tag, const SparseTensor &) override;
};

} // end namespace smurff