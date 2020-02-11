#pragma once

#include <string>
#include <memory>
#include <vector>
#include <set>
#include <cstdint>

#include <highfive/H5File.hpp>

#include <SmurffCpp/Types.h>

namespace h5 = HighFive;


namespace smurff {

   class Model;
   class Result;
   class ILatentPrior;
   class MatrixConfig;

   class StepFile : public std::enable_shared_from_this<StepFile>
   {
   private:
      std::int32_t m_isample;
      mutable h5::Group m_group;
      bool m_checkpoint;
      bool m_final;

   public:
      //this constructor should be used to create a step file on a first run of session
      StepFile(h5::Group group, std::int32_t isample, bool checkpoint);

      //this constructor should be used to  open existing step file when previous session is continued
      StepFile(h5::Group group);

   public:
      bool hasModel(std::uint64_t index) const;
      bool hasMu(std::uint64_t index) const;
      bool hasLinkMatrix(std::uint32_t mode) const;
      bool hasPred() const;

      std::shared_ptr<Matrix> getModel(std::uint64_t index) const;
      std::shared_ptr<Matrix> getMu(std::uint64_t index) const;
      std::shared_ptr<Matrix> getLinkMatrix(std::uint32_t mode) const;

      std::shared_ptr<Matrix> getPred() const;
      void getPredState(double &rmse_avg, double &rmse_1sample, double &auc_avg, double &auc_1sample, int &sample_iter, int &burnin_iter) const;
      std::shared_ptr<Matrix> getPredAvg() const;
      std::shared_ptr<Matrix> getPredVar() const;

      void putModel(const std::vector<std::shared_ptr<Matrix>> &) const;
      void putPostMuLambda(std::uint64_t index, const Matrix &, const Matrix &) const;
      void putMu(std::uint64_t index, const Matrix &) const;
      void putLinkMatrix(std::uint64_t mode, const Matrix &) const;

      void putPredState(double rmse_avg, double rmse_1sample, double auc_avg, double auc_1sample, int sample_iter, int burnin_iter) const;
      void putPredAvgVar(const SparseMatrix &, const SparseMatrix &, const SparseMatrix &) const;

   public:
      void save(std::shared_ptr<const Model> model, std::shared_ptr<const Result> pred, const std::vector<std::shared_ptr<ILatentPrior> >& priors) const;

   public:
      void restoreModel(std::shared_ptr<Model> model, int skip_mode = -1) const;
      void restorePred(std::shared_ptr<Result> m_pred) const;
      void restorePriors(std::vector<std::shared_ptr<ILatentPrior> >& priors) const;
      
      //-- used in PredictSession
      std::shared_ptr<Model> restoreModel(int skip_mode = -1) const;

      void restore(std::shared_ptr<Model> model, std::shared_ptr<Result> pred, std::vector<std::shared_ptr<ILatentPrior> >& priors) const;

   public:
      void remove(bool model, bool pred, bool priors) const;

   public:
      unsigned getNModes() const;
      std::int32_t getIsample() const;
      bool isCheckpoint() const;
      std::string getName() const;

   private:
      bool hasDataSet(const std::string &section, const std::string& tag) const;

      std::shared_ptr<Matrix> getMatrix(const std::string &section, const std::string& tag) const;
      std::shared_ptr<Vector> getVector(const std::string &section, const std::string& tag) const;
      std::shared_ptr<SparseMatrix> getSparseMatrix(const std::string &section, const std::string& tag) const;

      void putMatrix(const std::string &section, const std::string& tag, const Matrix &) const;
      void putSparseMatrix(const std::string &section, const std::string& tag, const SparseMatrix &) const;
   };
}
