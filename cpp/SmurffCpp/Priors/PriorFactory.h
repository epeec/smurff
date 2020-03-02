#pragma once

#include <SmurffCpp/Priors/IPriorFactory.h>
#include <SmurffCpp/Priors/ILatentPrior.h>
#include <SmurffCpp/Priors/MacauOnePrior.h>
#include <SmurffCpp/Priors/MacauPrior.h>
#include <SmurffCpp/Sessions/TrainSession.h>
#include <SmurffCpp/Configs/DataConfig.h>
#include <SmurffCpp/Configs/SideInfoConfig.h>

#include <SmurffCpp/SideInfo/DenseSideInfo.h>
#include <SmurffCpp/SideInfo/SparseSideInfo.h>

namespace smurff {

class PriorFactory : public IPriorFactory
{
public:
    template<class MacauPrior>
    std::shared_ptr<ILatentPrior> create_macau_prior(TrainSession &trainSession,
                                                     const std::shared_ptr<ISideInfo>& side_infos,
                                                     const SideInfoConfig& config_items);

    std::shared_ptr<ILatentPrior> create_macau_prior(TrainSession &trainSession, PriorTypes prior_type, 
                                                     const std::shared_ptr<ISideInfo>& side_infos,
                                                     const SideInfoConfig& config_items);

    template<class Factory>
    std::shared_ptr<ILatentPrior> create_macau_prior(TrainSession &trainSession, int mode, PriorTypes prior_type,
            const SideInfoConfig& config_items);

    std::shared_ptr<ILatentPrior> create_prior(TrainSession &trainSession, int mode) override;
};

//-------

template<class MacauPrior>
std::shared_ptr<ILatentPrior> PriorFactory::create_macau_prior(TrainSession &trainSession,
                                                               const std::shared_ptr<ISideInfo>& side_info,
                                                               const SideInfoConfig& config_item)
{
   std::shared_ptr<MacauPrior> prior(new MacauPrior(trainSession, -1));
   const auto& noise_config = config_item.getNoiseConfig();

   switch (noise_config.getNoiseType())
   {
   case NoiseTypes::fixed:
      {
         prior->addSideInfo(side_info, noise_config.getPrecision(), config_item.getTol(), config_item.getDirect(), false, config_item.getThrowOnCholeskyError());
      }
      break;
   case NoiseTypes::adaptive: // deprecated!
   case NoiseTypes::sampled:
      {
         prior->addSideInfo(side_info, noise_config.getPrecision(), config_item.getTol(), config_item.getDirect(), true, config_item.getThrowOnCholeskyError());
      }
      break;
   default:
      {
         THROWERROR("Unexpected noise type " + smurff::noiseTypeToString(noise_config.getNoiseType()) + " specified for macau prior. Allowed are: fixed and sampled noise.");
      }
   }

   return prior;
}

//mode - 0 (row), 1 (row)
//vsideinfo - vector of side feature configs (row or row)
template<class Factory>
std::shared_ptr<ILatentPrior> PriorFactory::create_macau_prior(TrainSession &trainSession, int mode, PriorTypes prior_type,
        const SideInfoConfig& config_item)
{
   Factory &subFactory = dynamic_cast<Factory &>(*this);

   std::shared_ptr<ISideInfo> side_info;
   if (config_item.isDense()) side_info = std::make_shared<DenseSideInfo>(config_item);
   else                       side_info = std::make_shared<SparseSideInfo>(config_item);

   return subFactory.create_macau_prior(trainSession, prior_type, side_info, config_item);
}

}
