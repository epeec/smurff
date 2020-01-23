#include "AdaptiveGaussianNoise.h"

#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/DataMatrices/Data.h>

namespace smurff {

double GaussianNoise::getAlpha() const
{
    return alpha;
}
} // end namespace smurff
