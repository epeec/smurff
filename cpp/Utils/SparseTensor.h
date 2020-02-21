#pragma once

#include <vector>
#include <iostream>
#include <memory>
#include <cstdint>
#include <algorithm>

#include <SmurffCpp/Utils/PVec.hpp>

namespace smurff
{
   struct SparseTensor
   {
      typedef std::vector<std::vector<std::uint32_t>> columns_type;
      std::vector<std::uint64_t> m_dims;
      columns_type               m_columns;
      std::vector<double>        m_values;

      int getNModes() const { return m_dims.size(); }
      const std::vector<std::uint64_t> & getDims() { return m_dims; };
      std::uint64_t getNNZ() const { return m_values.size(); }

      const std::vector<double>& getValues() const { return m_values; }
      const std::vector<std::uint32_t>& getColumn(int i) const { return m_columns.at(i); }

      std::vector<double>& getValues() { return m_values; }
      std::vector<std::uint32_t>& getColumn(int i)  { return m_columns.at(i); }

      std::pair<PVec<>, double> get(std::uint64_t) const;
      void set(std::uint64_t, PVec<>, double);
   };
}
