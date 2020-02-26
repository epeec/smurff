#include <SmurffCpp/result.h>

#include "ISession.h"

namespace smurff {

const std::vector<ResultItem>& ISession::getResultItems() const {
    return getResult().m_predictions;
}
} // end namespace smurff
