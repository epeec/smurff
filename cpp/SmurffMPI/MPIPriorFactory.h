#pragma once

#include <memory>

#include <SmurffCpp/Priors/PriorFactory.h>

namespace smurff {

   class ILatentPrior;
   class TrainSession;

   class MPIPriorFactory : public PriorFactory
   {
   public:
      std::shared_ptr<ILatentPrior> create_macau_prior(TrainSession &trainSession, PriorTypes prior_type,
                                                       const std::shared_ptr<ISideInfo>& side_info,
                                                       const SideInfoConfig& config_item);

      std::shared_ptr<ILatentPrior> create_prior(TrainSession &trainSession, int mode) override;
   };
}
