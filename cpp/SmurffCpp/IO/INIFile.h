// this code is based on https://github.com/Blandinium/inih/blob/master/cpp/INIReader.h 61bf1b3  on Dec 18, 2014

// Read an INI file into easy-to-access name/value pairs.

// inih and INIReader are released under the New BSD license (see LICENSE.txt).
// Go to the project home page for more info: http://code.google.com/p/inih/

//This code is heavily changed to support our needs

#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include <boost/property_tree/ptree.hpp>

namespace pt = boost::property_tree;

class INIFile
{
private:
   std::string m_filePath;
   bool m_modified;
   pt::ptree m_tree;

public:
    INIFile();
    ~INIFile();

public:
   void open(const std::string& filename);
   void create(const std::string& filename);

public:
   template<typename T>
   T get(const std::string& section, const std::string& name, const T& default_value) const
   {
      return m_tree.get<T>(section + "." + name, default_value);
   }

public:
    // Returns true is section with name exists
    bool hasSection(const std::string &name) const;

public:
   //appends item to the end of file - this is not possible to easily insert item in the arbitrary section so it is not supported - write is buffered
   void appendItem(const std::string& section, const std::string& tag, const std::string& value);

   //flushes write buffer to file
   void flush();
};
