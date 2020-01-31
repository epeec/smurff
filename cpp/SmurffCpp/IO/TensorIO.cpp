#include "TensorIO.h"

#include <fstream>
#include <algorithm>
#include <numeric>

#include <Utils/Error.h>

#include <SmurffCpp/IO/GenericIO.h>

namespace smurff {

#define EXTENSION_SDT ".sdt" //sparse double tensor (binary file)
#define EXTENSION_SBT ".sbt" //sparse binary tensor (binary file)
#define EXTENSION_TNS ".tns" //sparse tensor (txt file)
#define EXTENSION_CSV ".csv" //dense tensor (txt file)
#define EXTENSION_DDT ".ddt" //dense double tensor (binary file)

tensor_io::TensorType tensor_io::ExtensionToTensorType(const std::string& fname)
{
   std::size_t dotIndex = fname.find_last_of(".");
   if (dotIndex == std::string::npos)
   {
      THROWERROR("Extension is not specified in " + fname);
   }

   std::string extension = fname.substr(dotIndex);

   if (extension == EXTENSION_SDT)
   {
      return tensor_io::TensorType::sdt;
   }
   else if (extension == EXTENSION_SBT)
   {
      return tensor_io::TensorType::sbt;
   }
   else if (extension == EXTENSION_TNS)
   {
      return tensor_io::TensorType::tns;
   }
   else if (extension == EXTENSION_CSV)
   {
      return tensor_io::TensorType::csv;
   }
   else if (extension == EXTENSION_DDT)
   {
      return tensor_io::TensorType::ddt;
   }
   else
   {
      THROWERROR("Unknown file type: " + extension + " specified in " + fname);
   }
   return tensor_io::TensorType::none;
}

std::string TensorTypeToExtension(tensor_io::TensorType tensorType)
{
   switch (tensorType)
   {
   case tensor_io::TensorType::sdt:
      return EXTENSION_SDT;
   case tensor_io::TensorType::sbt:
      return EXTENSION_SBT;
   case tensor_io::TensorType::tns:
      return EXTENSION_TNS;
   case tensor_io::TensorType::csv:
       return EXTENSION_CSV;
   case tensor_io::TensorType::ddt:
      return EXTENSION_DDT;
   case tensor_io::TensorType::none:
      {
         THROWERROR("Unknown tensor type");
      }
   default:
      {
         THROWERROR("Unknown tensor type");
      }
   }
   return std::string();
}

std::shared_ptr<TensorConfig> tensor_io::read_tensor(const std::string& filename, bool isScarce)
{
   std::shared_ptr<TensorConfig> ret;

   TensorType tensorType = ExtensionToTensorType(filename);
   
   THROWERROR_FILE_NOT_EXIST(filename);

   switch (tensorType)
   {
   case tensor_io::TensorType::sdt:
      {
         std::ifstream fileStream(filename, std::ios_base::binary);
         THROWERROR_ASSERT_MSG(fileStream.is_open(), "Error opening file: " + filename);
         ret = tensor_io::read_sparse_float64_bin(fileStream, isScarce);
         break;
      }
   case tensor_io::TensorType::sbt:
      {
         std::ifstream fileStream(filename, std::ios_base::binary);
         THROWERROR_ASSERT_MSG(fileStream.is_open(), "Error opening file: " + filename);
         ret = tensor_io::read_sparse_binary_bin(fileStream, isScarce);
         break;
      }
   case tensor_io::TensorType::tns:
      {
         std::ifstream fileStream(filename);
         THROWERROR_ASSERT_MSG(fileStream.is_open(), "Error opening file: " + filename);
         ret = tensor_io::read_sparse_float64_tns(fileStream, isScarce);
         break;
      }
   case tensor_io::TensorType::csv:
      {
         std::ifstream fileStream(filename);
         THROWERROR_ASSERT_MSG(fileStream.is_open(), "Error opening file: " + filename);
         ret = tensor_io::read_dense_float64_csv(fileStream);
         break;
      }
   case tensor_io::TensorType::ddt:
      {
         std::ifstream fileStream(filename, std::ios_base::binary);
         THROWERROR_ASSERT_MSG(fileStream.is_open(), "Error opening file: " + filename);
         ret = tensor_io::read_dense_float64_bin(fileStream);
         break;
      }
   case tensor_io::TensorType::none:
      {
         THROWERROR("Unknown tensor type specified in " + filename);
      }
   default:
      {
         THROWERROR("Unknown tensor type specified in " + filename);
      }
   }

   ret->setFilename(filename);

   return ret;
}

std::shared_ptr<TensorConfig> tensor_io::read_dense_float64_bin(std::istream& in)
{
   std::uint64_t nmodes;
   in.read(reinterpret_cast<char*>(&nmodes), sizeof(std::uint64_t));

   std::vector<uint64_t> dims(nmodes);
   in.read(reinterpret_cast<char*>(dims.data()), dims.size() * sizeof(std::uint64_t));

   std::uint64_t nnz = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::uint64_t>());
   std::vector<double> values(nnz);
   in.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(double));

   return std::make_shared<TensorConfig>(dims, values.data(), NoiseConfig());
}

template<typename T>
static void read_line_single(std::istream& in, T &value)
{
   std::stringstream ss;
   std::string line;
   getline(in, line);
   ss << line;
   ss >> value;
}

template<typename T>
static void read_line_delim(std::istream& in, std::vector<T> &values, const char delim, std::uint64_t expected)
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

std::shared_ptr<TensorConfig> tensor_io::read_dense_float64_csv(std::istream &in)
{
   //nmodes
   std::uint64_t nmodes;
   read_line_single(in, nmodes);

   //dimensions
   std::vector<uint64_t> dims;
   read_line_delim(in, dims, ',', nmodes);
   std::uint64_t nnz = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::uint64_t>());

   //values
   std::vector<double> values;
   read_line_delim(in, values, ',', nnz);

   return std::make_shared<TensorConfig>(dims, values, NoiseConfig());
}

std::shared_ptr<TensorConfig> tensor_io::read_sparse_float64_bin(std::istream& in, bool isScarce)
{
   std::uint64_t nmodes;
   in.read(reinterpret_cast<char*>(&nmodes), sizeof(std::uint64_t));

   std::vector<uint64_t> dims(nmodes);
   in.read(reinterpret_cast<char*>(dims.data()), dims.size() * sizeof(std::uint64_t));

   std::uint64_t nnz;
   in.read(reinterpret_cast<char*>(&nnz), sizeof(std::uint64_t));

   TensorConfig::columns_type         column_vectors(nmodes);
   std::vector<const std::uint32_t *> column_ptrs;
   for (std::uint64_t i = 0; i < nmodes; i++)
   {
      auto &col = column_vectors.at(i);
      col.resize(nnz);
      in.read(reinterpret_cast<char*>(col.data()), nnz * sizeof(std::uint32_t));
      std::for_each(col.begin(), col.end(), [](std::uint32_t& c){ c--; });
      column_ptrs.push_back(col.data());
   }

   std::vector<double> values(nnz);
   in.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(double));

   return std::make_shared<TensorConfig>(dims, nnz, column_ptrs, values.data(), NoiseConfig(), isScarce);
}

std::shared_ptr<TensorConfig> tensor_io::read_sparse_float64_tns(std::istream& in, bool isScarce)
{
   std::stringstream ss;
   std::string line;
   std::string cell;

   // nmodes
   std::uint64_t nmodes;
   read_line_single(in, nmodes);
   
   //dimentions
   std::vector<uint64_t> dims;
   read_line_delim(in, dims, '\t', nmodes);

   // nnz
   std::uint64_t nnz;
   read_line_single(in, nnz);

   //columns
   TensorConfig::columns_type column_vectors(nmodes);
   for (std::uint64_t i = 0; i < nmodes; i++)
   {
      auto &col = column_vectors.at(i);
      read_line_delim(in, col, '\t', nnz);
      std::for_each(col.begin(), col.end(), [](std::uint32_t &c) { c--; });
   }

   //values
   std::vector<double> values;
   read_line_delim(in, values, '\t', nnz);

   return std::make_shared<TensorConfig>(dims, column_vectors, values, NoiseConfig(), isScarce);
}

std::shared_ptr<TensorConfig> tensor_io::read_sparse_binary_bin(std::istream& in, bool isScarce)
{
   std::uint64_t nmodes;
   in.read(reinterpret_cast<char*>(&nmodes), sizeof(std::uint64_t));

   std::vector<std::uint64_t> dims(nmodes);
   in.read(reinterpret_cast<char*>(dims.data()), dims.size() * sizeof(std::uint64_t));

   std::uint64_t nnz;
   in.read(reinterpret_cast<char*>(&nnz), sizeof(std::uint64_t));

   TensorConfig::columns_type         column_vectors(nmodes);
   for (std::uint64_t i = 0; i < nmodes; i++)
   {
      auto &col = column_vectors.at(i);
      col.resize(nnz);
      in.read(reinterpret_cast<char*>(col.data()), nnz * sizeof(std::uint32_t));
      std::for_each(col.begin(), col.end(), [](std::uint32_t& c){ c--; });
   }

   return std::make_shared<TensorConfig>(dims, column_vectors, NoiseConfig(), isScarce);
}

// ======================================================================================================

void tensor_io::write_tensor(const std::string& filename, std::shared_ptr<const TensorConfig> tensorConfig)
{
   TensorType tensorType = ExtensionToTensorType(filename);
   switch (tensorType)
   {
   case tensor_io::TensorType::sdt:
      {
         std::ofstream fileStream(filename, std::ios_base::binary);
         THROWERROR_ASSERT_MSG(fileStream.is_open(), "Error opening file: " + filename);
         tensor_io::write_sparse_float64_bin(fileStream, tensorConfig);
      }
      break;
   case tensor_io::TensorType::sbt:
      {
         std::ofstream fileStream(filename, std::ios_base::binary);
         THROWERROR_ASSERT_MSG(fileStream.is_open(), "Error opening file: " + filename);
         tensor_io::write_sparse_binary_bin(fileStream, tensorConfig);
      }
      break;
   case tensor_io::TensorType::tns:
      {
         std::ofstream fileStream(filename);
         THROWERROR_ASSERT_MSG(fileStream.is_open(), "Error opening file: " + filename);
         tensor_io::write_sparse_float64_tns(fileStream, tensorConfig);
      }
      break;
   case tensor_io::TensorType::csv:
      {
         std::ofstream fileStream(filename);
         THROWERROR_ASSERT_MSG(fileStream.is_open(), "Error opening file: " + filename);
         tensor_io::write_dense_float64_csv(fileStream, tensorConfig);
      }
      break;
   case tensor_io::TensorType::ddt:
      {
         std::ofstream fileStream(filename, std::ios_base::binary);
         THROWERROR_ASSERT_MSG(fileStream.is_open(), "Error opening file: " + filename);
         tensor_io::write_dense_float64_bin(fileStream, tensorConfig);
      }
      break;
   case tensor_io::TensorType::none:
      {
         THROWERROR("Unknown tensor type");
      }
   default:
      {
         THROWERROR("Unknown tensor type");
      }
   }
}

void tensor_io::write_dense_float64_bin(std::ostream& out, std::shared_ptr<const TensorConfig> tensorConfig)
{
   std::uint64_t nmodes = tensorConfig->getNModes();
   const std::vector<std::uint64_t>& dims = tensorConfig->getDims();
   const std::vector<double>& values = tensorConfig->getValues();

   out.write(reinterpret_cast<const char*>(&nmodes), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(dims.data()), dims.size() * sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(double));
}

template<typename T>
static void write_line_delim(std::ostream& out, const std::vector<T> &values, const std::string& delim)
{
   for(std::uint64_t i = 0; i < values.size(); i++)
      out << values[i] << (i < values.size() - 1 ? delim : "\n");
}

template<typename T>
static void write_line_delim_inc(std::ostream& out, const std::vector<T> &values, const std::string& delim)
{
   for(std::uint64_t i = 0; i < values.size(); i++)
      out << (values[i] + 1) << (i < values.size() - 1 ? delim : "\n");
}

void tensor_io::write_dense_float64_csv(std::ostream& out, std::shared_ptr<const TensorConfig> tensorConfig)
{
   out <<  tensorConfig->getNModes() << std::endl;
   write_line_delim(out, tensorConfig->getDims(), ",");
   write_line_delim(out, tensorConfig->getValues(), ",");
}

void tensor_io::write_sparse_float64_bin(std::ostream& out, std::shared_ptr<const TensorConfig> tensorConfig)
{
   write_sparse_binary_bin(out,  tensorConfig);
   const std::vector<double>& values = tensorConfig->getValues();
   out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(double));
}

void tensor_io::write_sparse_float64_tns(std::ostream& out, std::shared_ptr<const TensorConfig> tensorConfig)
{
   out << tensorConfig->getNModes() << std::endl;
   write_line_delim(out, tensorConfig->getDims(), "\t");
   out <<  tensorConfig->getNNZ() << std::endl;
   for(int i=0; i<tensorConfig->getNModes(); i++) write_line_delim_inc(out, tensorConfig->getColumn(i), "\t");
   write_line_delim(out, tensorConfig->getValues(), "\t");
}

void tensor_io::write_sparse_binary_bin(std::ostream& out, std::shared_ptr<const TensorConfig> tensorConfig)
{
   std::uint64_t nmodes = tensorConfig->getNModes();
   std::uint64_t nnz = tensorConfig->getNNZ();
   const std::vector<std::uint64_t>& dims = tensorConfig->getDims();

   out.write(reinterpret_cast<const char*>(&nmodes), sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(dims.data()), dims.size() * sizeof(std::uint64_t));
   out.write(reinterpret_cast<const char*>(&nnz), sizeof(std::uint64_t));

   for (int i = 0; i < nmodes; ++i)
   {
      std::vector<std::uint32_t> column = tensorConfig->getColumn(i); //create copy of column
      std::for_each(column.begin(), column.end(), [](std::uint32_t &col) { col++; });
      out.write(reinterpret_cast<const char*>(column.data()), column.size() * sizeof(std::uint32_t));
   }
}
} // end namespace smurff
