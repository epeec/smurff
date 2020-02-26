#pragma once

#include <string>
#include <memory>
#include <vector>
#include <set>
#include <cstdint>

#include <SmurffCpp/Types.h>
#include <SmurffCpp/Utils/HDF5.h>

namespace h5 = HighFive;

namespace smurff {

   class Model;
   class Result;
   class ILatentPrior;

   class Step : private HDF5
   {
   private:
      mutable h5::File m_file;
      std::int32_t m_isample;
      bool m_checkpoint;
      bool m_final;

   public:
      //this constructor should be used to create a step file on a first run of session
      Step(h5::File file, std::int32_t isample, bool checkpoint);

      //this constructor should be used to  open existing step file when previous session is continued
      Step(h5::File file, h5::Group group);

      ~Step();

   public:
      bool hasModel(std::uint64_t index) const;
      bool hasPred() const;

      void readModel(std::uint64_t index, Matrix &) const;
      void readMu(std::uint64_t index, Vector &) const;
      void readLinkMatrix(std::uint32_t index, Matrix &) const;

      void getPredState(double &rmse_avg, double &rmse_1sample, double &auc_avg, double &auc_1sample, int &sample_iter, int &burnin_iter) const;

      void readPred(Matrix &) const;
      void readPredAvg(Matrix &) const;
      void readPredVar(Matrix &) const;

      void putModel(const std::vector<Matrix> &);
      void putPostMuLambda(std::uint64_t index, const Matrix &, const Matrix &);

      void putMu(std::uint64_t index, const Matrix &);
      void putLinkMatrix(std::uint64_t mode, const Matrix &);

      void putPredState(double rmse_avg, double rmse_1sample, double auc_avg, double auc_1sample, int sample_iter, int burnin_iter);
      void putPredAvgVar(const SparseMatrix &, const SparseMatrix &, const SparseMatrix &);

   public:
      void save(const Model &model, std::shared_ptr<const Result> pred, const std::vector<std::shared_ptr<ILatentPrior> >& priors);

   public:
      void restore(Model &model, std::shared_ptr<Result> pred, std::vector<std::shared_ptr<ILatentPrior> >& priors) const;

   public:
      void remove(bool model, bool pred, bool priors);

   public:
      unsigned getNModes() const;
      std::int32_t getIsample() const;
      bool isCheckpoint() const;
      std::string getName() const;

   private:
      void readMatrix(const std::string &section, const std::string &tag, Matrix &) const;
      void readVector(const std::string &section, const std::string &tag, Vector &) const;
      std::shared_ptr<SparseMatrix> getSparseMatrix(const std::string &section, const std::string &tag) const;
   };
}
