#include <string>
#include <iostream>
#include <memory>

#include <SmurffCpp/Configs/TensorConfig.h>

namespace smurff { namespace tensor_io
{

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

   enum class TensorType
   {
      //sparse types
      none,
      sdt,
      sbt,
      tns,

      //dense types
      csv,
      ddt
   };

   TensorType ExtensionToTensorType(const std::string& fname);

   std::shared_ptr<TensorConfig> read_tensor(const std::string& filename, bool isScarce);

   std::shared_ptr<TensorConfig> read_dense_float64_bin(std::istream& in);
   std::shared_ptr<TensorConfig> read_dense_float64_csv(std::istream& in);

   std::shared_ptr<TensorConfig> read_sparse_float64_bin(std::istream& in, bool isScarce);
   std::shared_ptr<TensorConfig> read_sparse_float64_tns(std::istream& in, bool isScarce);

   std::shared_ptr<TensorConfig> read_sparse_binary_bin(std::istream& in, bool isScarce);

   // ===

   void write_tensor(const std::string& filename, std::shared_ptr<const TensorConfig> tensorConfig);

   void write_dense_float64_bin(std::ostream& out, std::shared_ptr<const TensorConfig> tensorConfig);
   void write_dense_float64_csv(std::ostream& out, std::shared_ptr<const TensorConfig> tensorConfig);

   void write_sparse_float64_bin(std::ostream& out, std::shared_ptr<const TensorConfig> tensorConfig);
   void write_sparse_float64_tns(std::ostream& out, std::shared_ptr<const TensorConfig> tensorConfig);

   void write_sparse_binary_bin(std::ostream& out, std::shared_ptr<const TensorConfig> tensorConfig);
}}
