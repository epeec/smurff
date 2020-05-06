#pragma once

#include <string>
#include <memory>
#include <vector>
#include <set>
#include <cstdint>

#include <SmurffCpp/Types.h>
#include <SmurffCpp/Utils/HDF5Group.h>

namespace h5 = HighFive;

namespace smurff {

   class Model;
   class Result;
   class ILatentPrior;

   class SaveState : private HDF5Group
   {
   private:
      h5::File m_file;
      std::int32_t m_isample;
      bool m_checkpoint;
      bool m_save_aggr;

   public:
      //this constructor should be used to create a step file on a first run of trainSession
      SaveState(h5::File file, std::int32_t isample, bool checkpoint, bool final);

      //this constructor should be used to  open existing step file when previous trainSession is continued
      SaveState(h5::File file, h5::Group group);

      ~SaveState();

   public:
      bool hasModel(std::uint64_t index) const;
      bool hasAggr(std::uint64_t index) const;
      bool hasPred() const;


      void readModel(std::uint64_t index, Matrix &) const;
      void readMu(std::uint64_t index, Vector &) const;
      void readLinkMatrix(std::uint32_t index, Matrix &) const;
      void readAggr(std::uint64_t index, int &, Matrix &, Matrix &) const;
      void readPostMuLambda(std::uint64_t index, Matrix &, Matrix &) const;

      void getPredState(double &rmse_avg, double &rmse_1sample, double &auc_avg, double &auc_1sample, int &sample_iter, int &burnin_iter) const;

      void readPred(Matrix &) const;
      void readPredAvg(Matrix &) const;
      void readPredVar(Matrix &) const;

      void putModel(const std::vector<Matrix> &);
      void putAggr(std::uint64_t index, const int, const Matrix &, const Matrix &);
      void putPostMuLambda(std::uint64_t index, const Matrix &, const Matrix &);

      void putMu(std::uint64_t index, const Matrix &);
      void putLinkMatrix(std::uint64_t mode, const Matrix &);

      void putPredState(double rmse_avg, double rmse_1sample, double auc_avg, double auc_1sample, int sample_iter, int burnin_iter);
      void putPredAvgVar(const SparseMatrix &, const SparseMatrix &, const SparseMatrix &);

   public:
      unsigned     getNModes() const;
      std::int32_t getIsample() const;
      bool         isCheckpoint() const;
      bool         saveAggr() const;
      std::string  getName() const;
   };
}
