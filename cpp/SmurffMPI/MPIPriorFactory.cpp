#include "MPIPriorFactory.h"

#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Priors/ILatentPrior.h>
#include <SmurffMPI/MPIPriorFactory.h>
#include <SmurffMPI/MPIMacauPrior.h>

namespace smurff {

std::shared_ptr<ILatentPrior> MPIPriorFactory::create_macau_prior(TrainSession &trainSession, PriorTypes prior_type, 
                                                                  const std::shared_ptr<ISideInfo>& side_info,
                                                                  const SideInfoConfig& config_item)
{
   return PriorFactory::create_macau_prior<MPIMacauPrior>(trainSession, side_info, config_item);
}

std::shared_ptr<ILatentPrior> MPIPriorFactory::create_prior(TrainSession &trainSession, int mode)
{
   PriorTypes pt = trainSession.getConfig().getPriorTypes().at(mode);

   if(pt == PriorTypes::macau)
   {
      return PriorFactory::create_macau_prior<MPIPriorFactory>(trainSession, mode, pt, trainSession.getConfig().getSideInfoConfig(mode));
   }
   else
   {
      return PriorFactory::create_prior(trainSession, mode);
   }
}
} // end namespace smurff
