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
    // Get a string value from INI file, 
    // returning default_value if not found.
    std::string getString(const std::string& section, const std::string& name, const std::string& default_value) const;

    // Get an integer (long) value from INI file, 
    // returning default_value if not found or not a valid integer (decimal "1234", "-1234", or hex "0x4d2").
    int getInteger(const std::string& section, const std::string& name, int default_value) const;

    // Get a real (floating point double) value from INI file, 
    // returning default_value if not found or not a valid floating point value according to strtod().
    double getReal(const std::string& section, const std::string& name, double default_value) const;

    // Get a boolean value from INI file, 
    // returning default_value if not found or if not a valid true/false value. 
    // Valid true values are "true", "yes", "on", "1",
    // and valid false values are "false", "no", "off", "0" (not case sensitive).
    bool getBoolean(const std::string& section, const std::string& name, bool default_value) const;

public:
    // Returns true is section with name exists
    bool hasSection(const std::string &name) const;

public:
   //appends item to the end of file - this is not possible to easily insert item in the arbitrary section so it is not supported - write is buffered
   void appendItem(const std::string& section, const std::string& tag, const std::string& value);

   //flushes write buffer to file
   void flush();
};
