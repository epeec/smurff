#include "Tensor.h"

#include <numeric>

#include <SmurffCpp/Utils/PVec.hpp>

namespace smurff {

std::pair<PVec<>, double> SparseTensor::get(std::uint64_t pos) const
{
   double val = m_values.at(pos);
   PVec<> coords(getNModes());
   for (unsigned j = 0; j < getNModes(); ++j)
         coords[j] = m_columns[j][pos];

   return std::make_pair(PVec<>(coords), val);
}

void SparseTensor::set(std::uint64_t pos, PVec<> coords, double value)
{
    m_values[pos] = value;
    for(unsigned j=0; j<getNModes(); ++j) m_columns[j][pos] = coords[j];
}

} // end namespace smurff
