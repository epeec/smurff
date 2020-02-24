#pragma once

#include <vector>
#include <iostream>
#include <memory>
#include <cstdint>
#include <algorithm>

#include <SmurffCpp/Utils/PVec.hpp>

namespace smurff
{
   struct Tensor
   {
      Tensor() {}

      Tensor(
          const std::vector<std::uint64_t> &dims,
          const std::vector<double> &values)
         : m_dims(dims), m_values(values) {}

      std::vector<std::uint64_t> m_dims;
      std::vector<double>        m_values;

      int getNModes() const { return m_dims.size(); }
      const std::vector<std::uint64_t> & getDims() const { return m_dims; };
      const std::uint64_t & getNRow() const { return m_dims.at(0); };
      const std::uint64_t & getNCol() const { return m_dims.at(1); };
      std::uint64_t getNNZ() const { return m_values.size(); }

      const std::vector<double>& getValues() const { return m_values; }
      std::vector<double>& getValues() { return m_values; }
   };

   struct SparseTensor : public Tensor
   {
      typedef std::vector<std::vector<std::uint32_t>> columns_type;

      SparseTensor() {}

      SparseTensor(
          const std::vector<std::uint64_t> &dims,
          const columns_type &columns,
          const std::vector<double> &values)
      : Tensor(dims, values), m_columns(columns) {}

      columns_type m_columns;

      const std::vector<std::uint32_t> &getRows() const { return m_columns.at(0); }
      std::vector<std::uint32_t> &getRows() { return m_columns.at(0); }

      const std::vector<std::uint32_t> &getCols() const { return m_columns.at(1); }
      std::vector<std::uint32_t> &getCols() { return m_columns.at(1); }

      const std::vector<std::uint32_t> &getColumn(int i) const { return m_columns.at(i); }
      std::vector<std::uint32_t> &getColumn(int i) { return m_columns.at(i); }

      std::pair<PVec<>, double> get(std::uint64_t) const;
      void set(std::uint64_t, PVec<>, double);
   };
}
