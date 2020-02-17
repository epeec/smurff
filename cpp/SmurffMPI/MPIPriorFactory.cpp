#include "MPIPriorFactory.h"

#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Priors/ILatentPrior.h>
#include <SmurffMPI/MPIPriorFactory.h>
#include <SmurffMPI/MPIMacauPrior.h>

namespace smurff {

std::shared_ptr<ILatentPrior> MPIPriorFactory::create_macau_prior(std::shared_ptr<Session> session, PriorTypes prior_type, 
                                                                  const std::shared_ptr<ISideInfo>& side_info,
                                                                  const std::shared_ptr<SideInfoConfig>& config_item)
{
   return PriorFactory::create_macau_prior<MPIMacauPrior>(session, side_info, config_item);
}

std::shared_ptr<ILatentPrior> MPIPriorFactory::create_prior(std::shared_ptr<Session> session, int mode)
{
   PriorTypes pt = session->getConfig().getPriorTypes().at(mode);

   if(pt == PriorTypes::macau)
   {
      return PriorFactory::create_macau_prior<MPIPriorFactory>(session, mode, pt, session->getConfig().getSideInfoConfig(mode));
   }
   else
   {
      return PriorFactory::create_prior(session, mode);
   }
}
} // end namespace smurff
