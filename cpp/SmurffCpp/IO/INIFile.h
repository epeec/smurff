#pragma once

#include <string>
#include <boost/property_tree/ptree_fwd.hpp>

namespace pt = boost::property_tree;

#include <SmurffCpp/Utils/ConfigFile.h>

namespace smurff 
{
class INIFile : public ConfigFile
{
public:
   INIFile();
   ~INIFile();

private:
   std::unique_ptr<pt::ptree> m_tree;
   pt::ptree makeUnique(const pt::ptree &pt);

public:
   void read(const std::string &filename);
   void write(const std::string &filename);

private:
   template <typename T>
   T getInternal(const std::string &section, const std::string &name, const T &default_value) const;

   template <typename T>
   void putInternal(const std::string &section, const std::string &tag, const T &value);

public:
   bool hasDataSet(const std::string &section, const std::string &tag) const override;
   bool hasSection(const std::string &section) const override;

   int get(const std::string &section, const std::string &tag, const int &default_value) const override;
   size_t get(const std::string &section, const std::string &tag, const size_t &default_value) const override;
   double get(const std::string &section, const std::string &tag, const double &default_value) const override;
   bool get(const std::string &section, const std::string &tag, const bool &default_value) const override;
   std::string get(const std::string &section, const std::string &tag, const std::string &default_value) const override;
   void read(const std::string &section, const std::string &tag, Vector &) const override;
   void read(const std::string &section, const std::string &tag, Matrix &) const override;
   void read(const std::string &section, const std::string &tag, SparseMatrix &) const override;
   void read(const std::string &section, const std::string &tag, DenseTensor &) const override;
   void read(const std::string &section, const std::string &tag, SparseTensor &) const override;

   void put(const std::string &section, const std::string &tag, const int &value) override;
   void put(const std::string &section, const std::string &tag, const size_t &value) override;
   void put(const std::string &section, const std::string &tag, const double &value) override;
   void put(const std::string &section, const std::string &tag, const bool &value) override;
   void put(const std::string &section, const std::string &tag, const std::string &value) override;

   void write(const std::string &section, const std::string &tag, const Vector &) override;
   void write(const std::string &section, const std::string &tag, const Matrix &) override;
   void write(const std::string &section, const std::string &tag, const SparseMatrix &) override;
   void write(const std::string &section, const std::string &tag, const DenseTensor &) override;
   void write(const std::string &section, const std::string &tag, const SparseTensor &) override;
};

} // end namespace smurff