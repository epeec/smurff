#include <mpi.h>

#include "MPISession.h"
#include "MPIPriorFactory.h"

#include <SmurffCpp/Priors/ILatentPrior.h>

#include <Utils/Error.h>

namespace smurff {

MPISession::MPISession()
{
   name = "MPISession";

   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
}

MPISession::MPISession(const Config &config)
    : TrainSession(config)
{
   name = "MPISession";

   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
}

void MPISession::run()
{
   if (world_rank == 0)
   {
      TrainSession::run();
   }
   else
   {
      bool work_done = false;

      for(auto &p : m_priors)
         work_done |= p->run_slave();

      THROWERROR_ASSERT(work_done);
   }
}

std::shared_ptr<IPriorFactory> MPISession::create_prior_factory() const
{
   return std::make_shared<MPIPriorFactory>();
}

//create mpi trainSession
//parses args with setFromArgs, then internally calls setFromConfig (to validate, save, set config)
std::shared_ptr<ISession> create_mpi_session(int argc, char** argv)
{
   Config config  = parse_options(argc, argv);
   return std::make_shared<MPISession>(config);
}
} // end namespace smurff
