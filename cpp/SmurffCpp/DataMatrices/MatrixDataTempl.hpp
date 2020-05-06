#pragma once

#include <memory>

#include "MatrixData.h"

#include <SmurffCpp/Utils/Error.h>

namespace smurff
{
   template<typename YType>
   class MatrixDataTempl : public MatrixData
   {
   private:
      // matrices with the data
      std::vector<YType> m_Yv;

   public:
      MatrixDataTempl(YType Y)
      {
         m_Yv.push_back(Y);
         m_Yv.push_back(Y.transpose());
      }

      void init_pre() override
      {
         THROWERROR_ASSERT(nrow() > 0 && ncol() > 0);
      }

      PVec<> dim() const override 
      { 
         return PVec<>({ static_cast<int>(Y().rows()), static_cast<int>(Y().cols()) }); 
      }

      std::uint64_t nnz() const override 
      { 
         return Y().nonZeros(); 
      }

      double sum() const override 
      { 
         return Y().sum(); 
      }

   public:
      const YType& Y(int mode = 0) const
      {
         return m_Yv.at(mode);
      }
   };
}
