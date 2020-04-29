#include "NoiseFactory.h"

#include <SmurffCpp/Noises/AdaptiveGaussianNoise.h>
#include <SmurffCpp/Noises/FixedGaussianNoise.h>
#include <SmurffCpp/Noises/SampledGaussianNoise.h>
#include <SmurffCpp/Noises/ProbitNoise.h>
#include <SmurffCpp/Noises/UnusedNoise.h>

#include <SmurffCpp/Utils/Error.h>

namespace smurff {

std::unique_ptr<INoiseModel> NoiseFactory::create_noise_model(const NoiseConfig& config)
{
   switch(config.getNoiseType())
   {
      case NoiseTypes::fixed:
         return std::unique_ptr<INoiseModel>(new FixedGaussianNoise(config.getPrecision()));
      case NoiseTypes::sampled:
         return std::unique_ptr<INoiseModel>(new SampledGaussianNoise(config.getPrecision()));
      case NoiseTypes::adaptive:
         return std::unique_ptr<INoiseModel>(new AdaptiveGaussianNoise(config.getSnInit(), config.getSnMax()));
      case NoiseTypes::probit:
         return std::unique_ptr<INoiseModel>(new ProbitNoise(config.getThreshold()));
      case NoiseTypes::unused:
         return std::unique_ptr<INoiseModel>(new UnusedNoise());
      default:
      {
         THROWERROR("Unknown noise model type: " + std::to_string((int)config.getNoiseType()));
      }
   }
}
} // end namespace smurff
