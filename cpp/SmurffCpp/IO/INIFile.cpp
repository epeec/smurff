// Read an INI file into easy-to-access name/value pairs.
// this code is based on https://github.com/Blandinium/inih/blob/master/cpp/INIReader.cpp 61bf1b3  on Dec 18, 2014

#include <iostream>
#include <fstream>
#include <algorithm>

#include "INIFile.h"

#include <Utils/Error.h>

#include <boost/property_tree/ini_parser.hpp>

INIFile::INIFile()
   : m_modified(false)
{
}

INIFile::~INIFile()
{
   if (m_modified) flush();
}

void INIFile::open(const std::string& filename)
{
   m_filePath = filename;
   std::ifstream file;
   file.open(filename, std::ios::in);
   pt::ini_parser::read_ini(file, m_tree);
}

void INIFile::create(const std::string& filename)
{
   m_filePath = filename;
   m_tree.clear();
}

void INIFile::flush()
{
   std::ofstream file;
   file.open(m_filePath, std::ios::trunc);
   pt::ini_parser::write_ini(file, m_tree);
}

bool INIFile::hasSection(const std::string &name) const
{
   return m_tree.get_child_optional(name) != boost::none;
}

void INIFile::appendItem(const std::string& section, const std::string& tag, const std::string& value)
{
   m_modified = true;
   m_tree.put(section + "." + tag, value);
}
