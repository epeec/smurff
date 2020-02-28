#include <fstream>

#include <Utils/Error.h>

#include "INIFile.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

namespace smurff
{

INIFile::INIFile() : m_tree(std::make_unique<pt::ptree>()) {}

INIFile::~INIFile() {}

pt::ptree INIFile::makeUnique(const pt::ptree &pt)
{
   if (pt.size() <= 1)
      return pt;

   std::map<pt::ptree::key_type, size_t> counts;
   for (const auto &el : pt)
      counts[el.first]++;
   // remove counts of 1 or less
   for (auto &c : counts)
      if (c.second <= 1)
         c.second = 0;

   pt::ptree ret(pt.data());
   for (const auto &el : pt)
   {
      pt::ptree::key_type key = el.first;
      if (counts[key] > 0)
      {
         counts[key]--;
         key += "_" + std::to_string(counts[key]);
      }
      ret.push_back({key, makeUnique(el.second)});
   }

   return ret;
}

void INIFile::read(const std::string &filename)
{
   std::ifstream file;
   file.open(filename, std::ios::in);
   pt::ini_parser::read_ini(file, *m_tree);
}

void INIFile::write(const std::string &filename)
{
   std::ofstream file;
   file.open(filename, std::ios::trunc);
   pt::ini_parser::write_ini(file, makeUnique(*m_tree));
}

bool INIFile::hasSection(const std::string &name) const
{
   return m_tree->get_child_optional(name) != boost::none;
}

bool INIFile::hasDataSet(const std::string &name, const std::string &tag) const
{
   return false;
}

template <typename T>
T INIFile::getInternal(const std::string &section, const std::string &name, const T &default_value) const
{
   return m_tree->get<T>(section + "." + name, default_value);
}

int INIFile::get(const std::string &section, const std::string &tag, const int &default_value) const
{
   return getInternal(section, tag, default_value);
}

size_t INIFile::get(const std::string &section, const std::string &tag, const size_t &default_value) const
{
   return getInternal(section, tag, default_value);
}

double INIFile::get(const std::string &section, const std::string &tag, const double &default_value) const
{
   return getInternal(section, tag, default_value);
}

bool INIFile::get(const std::string &section, const std::string &tag, const bool &default_value) const
{
   return getInternal(section, tag, default_value);
}

std::string INIFile::get(const std::string &section, const std::string &tag, const std::string &default_value) const
{
   return getInternal(section, tag, default_value);
}

void INIFile::read(const std::string &section, const std::string &tag, Vector &) const
{
   THROWERROR_NOTIMPL();
}

void INIFile::read(const std::string &section, const std::string &tag, Matrix &) const
{
   THROWERROR_NOTIMPL();
}

void INIFile::read(const std::string &section, const std::string &tag, SparseMatrix &) const
{
   THROWERROR_NOTIMPL();
}

void INIFile::read(const std::string &section, const std::string &tag, DenseTensor &) const
{
   THROWERROR_NOTIMPL();
}

void INIFile::read(const std::string &section, const std::string &tag, SparseTensor &) const
{
   THROWERROR_NOTIMPL();
}

template <typename T>
void INIFile::putInternal(const std::string &section, const std::string &tag, const T &value)
{
   m_tree->put(section + "." + tag, value);
}

void INIFile::put(const std::string &section, const std::string &tag, const int &value)
{
   putInternal(section, tag, value);
}

void INIFile::put(const std::string &section, const std::string &tag, const size_t &value)
{
   putInternal(section, tag, value);
}

void INIFile::put(const std::string &section, const std::string &tag, const double &value)
{
   putInternal(section, tag, value);
}

void INIFile::put(const std::string &section, const std::string &tag, const bool &value)
{
   putInternal(section, tag, value);
}

void INIFile::put(const std::string &section, const std::string &tag, const std::string &value)
{
   putInternal(section, tag, value);
}

void INIFile::write(const std::string &section, const std::string &tag, const Vector &M)
{
}

void INIFile::write(const std::string &section, const std::string &tag, const Matrix &M)
{
}

void INIFile::write(const std::string &section, const std::string &tag, const SparseMatrix &X)
{
}

void INIFile::write(const std::string &section, const std::string &tag, const DenseTensor &X)
{
}

void INIFile::write(const std::string &section, const std::string &tag, const SparseTensor &X)
{
}

} //end namespace smurff