#pragma once

#include <SmurffCpp/Sessions/TrainSession.h>

namespace smurff {

class PythonSession : public TrainSession
{

private:
   static bool keepRunning;
   static bool keepRunningVerbose;

public:
   PythonSession(const Config &c)
   : TrainSession(c)
   {
      name = "PythonSession";
      keepRunning = true;
   }

public:
   bool interrupted() override
   {
       return !keepRunning;
   }

   bool step() override;

private:
   static void intHandler(int);
};

}
