#include "PriorFactory.h"

#include <SmurffCpp/Types.h>

#include <SmurffCpp/Priors/NormalPrior.h>
#include <SmurffCpp/Priors/NormalOnePrior.h>
#include <SmurffCpp/Priors/SpikeAndSlabPrior.h>

#include <Utils/Error.h>

#include <SmurffCpp/SideInfo/DenseSideInfo.h>
#include <SmurffCpp/SideInfo/SparseSideInfo.h>

#include <SmurffCpp/Utils/MatrixUtils.h>

namespace smurff {

//create macau prior features

//-------

std::shared_ptr<ILatentPrior> PriorFactory::create_macau_prior(TrainSession &trainSession, PriorTypes prior_type,
   const std::shared_ptr<ISideInfo>& side_info,
   const SideInfoConfig& config_item)
{
   if(prior_type == PriorTypes::macau || prior_type == PriorTypes::default_prior)
   {
      return create_macau_prior<MacauPrior>(trainSession, side_info, config_item);
   }
   else if(prior_type == PriorTypes::macauone)
   {
      return create_macau_prior<MacauOnePrior>(trainSession, side_info, config_item);
   }
   else
   {
      THROWERROR("Unknown prior with side info: " + priorTypeToString(prior_type));
   }
}

std::shared_ptr<ILatentPrior> PriorFactory::create_prior(TrainSession &trainSession, int mode)
{
   PriorTypes priorType = trainSession.getConfig().getPriorTypes().at(mode);

   switch(priorType)
   {
   case PriorTypes::normal:
   case PriorTypes::default_prior:
      return std::shared_ptr<NormalPrior>(new NormalPrior(trainSession, -1));
   case PriorTypes::spikeandslab:
      return std::shared_ptr<SpikeAndSlabPrior>(new SpikeAndSlabPrior(trainSession, -1));
   case PriorTypes::normalone:
      return std::shared_ptr<NormalOnePrior>(new NormalOnePrior(trainSession, -1));
   case PriorTypes::macau:
   case PriorTypes::macauone:
      return create_macau_prior<PriorFactory>(trainSession, mode, priorType, trainSession.getConfig().getSideInfoConfig(mode));
   default:
      {
         THROWERROR("Unknown prior: " + priorTypeToString(priorType));
      }
   }
}
} // end namespace smurff
