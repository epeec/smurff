// Read an INI file into easy-to-access name/value pairs.
// this code is based on https://github.com/Blandinium/inih/blob/master/cpp/INIReader.cpp 61bf1b3  on Dec 18, 2014

#include <fstream>

#include "INIFile.h"

#include <boost/property_tree/ini_parser.hpp>

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
   pt::ini_parser::write_ini(file, m_tree);
}

bool INIFile::hasSection(const std::string &name) const
{
   return m_tree.get_child_optional(name) != boost::none;
}