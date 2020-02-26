#pragma once

#include <memory>

namespace smurff {
   
class Session;
class ILatentPrior;

class IPriorFactory
{
public:
   virtual ~IPriorFactory()
   {
   }

public:
   virtual std::shared_ptr<ILatentPrior> create_prior(Session &session, int mode) = 0;
};
   
}
