#pragma once

#include <string>
#include <memory>
#include <vector>
#include <set>
#include <cstdint>

#include <highfive/H5File.hpp>

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
      h5::Group m_group;
      bool m_checkpoint;
      bool m_final;

   public:
      //this constructor should be used to create a step file on a first run of session
      StepFile(std::int32_t isample, h5::Group group, bool checkpoint, bool final);

   public:
      bool hasModel(std::uint64_t index) const;
      bool hasMu(std::uint64_t index) const;
      bool hasLinkMatrix(std::uint32_t mode) const;
      bool hasPred() const;

      h5::DataSet getModelDataSet(std::uint64_t index) const;
      h5::DataSet getMuDataSet(std::uint64_t index) const;
      h5::DataSet getLinkMatrixDataSet(std::uint32_t mode) const;
      h5::DataSet getPredDataSet() const;
      h5::DataSet getPredStateDataSet() const;
      h5::DataSet getPredAvgDataSet() const;
      h5::DataSet getPredVarDataSet() const;

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
      std::int32_t getIsample() const;
      bool isCheckpoint() const;

   public:
      bool hasDataSet(const std::string &section, const std::string& tag) const;
      h5::DataSet getDataSet(const std::string &section, const std::string& tag) const;
   };
}
