#pragma once

#include <string>
#include <iostream>
#include <memory>

#include <SmurffCpp/Configs/TensorConfig.h>

namespace smurff { namespace generic_io 
{
   std::shared_ptr<TensorConfig> read_data_config(const std::string& filename, bool isScarce);

   void write_data_config(const std::string& filename, std::shared_ptr<TensorConfig> tensorConfig);

   bool file_exists(const std::string& filepath);


   template<typename T>
   void write_line_delim(std::ostream& out, const std::vector<T> &values, const std::string& delim)
   {
      for(std::uint64_t i = 0; i < values.size(); i++)
         out << values[i] << (i < values.size() - 1 ? delim : "\n");
   }

   template<typename T>
   void write_line_delim_inc(std::ostream& out, const std::vector<T> &values, const std::string& delim)
   {
      for(std::uint64_t i = 0; i < values.size(); i++)
         out << (values[i] + 1) << (i < values.size() - 1 ? delim : "\n");
   }

   template<typename T>
   void read_line_single(std::istream& in, T &value)
   {
      std::stringstream ss;
      std::string line;
      getline(in, line);
      ss << line;
      ss >> value;
   }

   template<typename T>
   void read_line_delim(std::istream& in, std::vector<T> &values, const char delim, std::uint64_t expected)
   {
      std::stringstream ss;
      std::string line;
      std::string cell;
      T value;

      getline(in, line);

      std::stringstream lineStream0(line);

      std::uint64_t count = 0;
      while (std::getline(lineStream0, cell, delim))
      {
         ss.clear();
         ss << cell;
         ss >> value;
         values.push_back(value);
         count++;
      }

      THROWERROR_ASSERT_MSG(count == expected, "unexpected number of elements on line");
   }   
}}
