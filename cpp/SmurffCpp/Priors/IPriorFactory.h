#pragma once

#include <memory>

namespace smurff {
   
class TrainSession;
class ILatentPrior;

class IPriorFactory
{
public:
   virtual ~IPriorFactory()
   {
   }

public:
   virtual std::shared_ptr<ILatentPrior> create_prior(TrainSession &trainSession, int mode) = 0;
};
   
}
