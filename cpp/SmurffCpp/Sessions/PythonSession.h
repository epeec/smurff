#pragma once

#include <SmurffCpp/Sessions/TrainSession.h>

namespace smurff {

class SessionFactory;

class PythonSession : public TrainSession
{
   friend class SessionFactory;

private:
   static bool keepRunning;
   static bool keepRunningVerbose;

protected:
   PythonSession()
   {
      name = "PythonSession";
      keepRunning = true;
   }

protected:
   bool interrupted() override
   {
       return !keepRunning;
   }

   bool step() override;

private:
   static void intHandler(int);
};

}
