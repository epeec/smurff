#pragma once

#include <memory>
#include <string>

namespace smurff
{
   class Data;
   class MatrixConfig;
   class TensorConfig;
   class DataConfig;

   class IDataCreator
   {
   public:
      virtual ~IDataCreator(){}

   public:
      virtual std::shared_ptr<Data> create(std::shared_ptr<const DataConfig> dc) const = 0;
   };
}
