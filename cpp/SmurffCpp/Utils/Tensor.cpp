#include "Tensor.h"

#include <numeric>

#include <SmurffCpp/Utils/PVec.hpp>

namespace smurff {

std::pair<PVec<>, double> SparseTensor::get(std::uint64_t pos) const
{
   double val = getValues().at(pos);
   PVec<> coords(getNModes());
   for (unsigned j = 0; j < getNModes(); ++j)
         coords[j] = getColumn(j)[pos];

   return std::make_pair(PVec<>(coords), val);
}

void SparseTensor::set(std::uint64_t pos, PVec<> coords, double value)
{
    getValues()[pos] = value;
    for(unsigned j=0; j<getNModes(); ++j) getColumn(j)[pos] = coords[j];
}

} // end namespace smurff
