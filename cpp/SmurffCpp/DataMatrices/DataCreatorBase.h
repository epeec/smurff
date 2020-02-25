#pragma once

#include <memory>

#include "IDataCreator.h"

namespace smurff {
   class DataCreatorBase : public IDataCreator
   {
   public:
      DataCreatorBase()
      {
      }

   public:
      std::shared_ptr<Data> create(std::shared_ptr<const DataConfig> dc) const override;
   };
}
