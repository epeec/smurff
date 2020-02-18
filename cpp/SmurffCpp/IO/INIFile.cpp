// Read an INI file into easy-to-access name/value pairs.
// this code is based on https://github.com/Blandinium/inih/blob/master/cpp/INIReader.cpp 61bf1b3  on Dec 18, 2014

#include <fstream>
#include <map>

#include "INIFile.h"

#include <boost/property_tree/ini_parser.hpp>

pt::ptree INIFile::makeUnique(const pt::ptree &pt)
{
   if (pt.size() <= 1)
      return pt;

   std::map<pt::ptree::key_type, size_t> counts;
   for (const auto &el : pt) counts[el.first]++;
   // remove counts of 1 or less
   for (auto &c : counts) if (c.second <= 1) c.second = 0;


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

void INIFile::read(const std::string& filename)
{
   std::ifstream file;
   file.open(filename, std::ios::in);
   pt::ini_parser::read_ini(file, m_tree);
}

void INIFile::write(const std::string &filename)
{
   std::ofstream file;
   file.open(filename, std::ios::trunc);
   pt::ini_parser::write_ini(file, makeUnique(m_tree));
}

bool INIFile::hasSection(const std::string &name) const
{
   return m_tree.get_child_optional(name) != boost::none;
}

const pt::ptree &INIFile::getSection(const std::string &name) const
{
   return m_tree.get_child(name);
}

pt::ptree &INIFile::addSection(const std::string &name)
{
   return m_tree.add_child(name, pt::ptree());
}