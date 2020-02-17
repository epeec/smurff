#include <iostream>

#include <SmurffCpp/Utils/Step.h>

#include <SmurffCpp/Model.h>
#include <SmurffCpp/result.h>
#include <SmurffCpp/Priors/ILatentPrior.h>

#include <Utils/Error.h>
#include <Utils/StringUtils.h>
#include <SmurffCpp/IO/GenericIO.h>
#include <SmurffCpp/IO/MatrixIO.h>


#define LATENTS_SEC_TAG "latents"
#define PRED_SEC_TAG "predictions"
#define LINK_MATRICES_SEC_TAG "link_matrices"

#define IS_CHECKPOINT_TAG "is_checkpoint"
#define NUMBER_TAG "number"
#define NUM_MODES_TAG "num_modes"
#define PRED_TAG "pred"
#define PRED_STATE_TAG "pred_state"
#define PRED_AVG_TAG "pred_avg"
#define PRED_VAR_TAG "pred_var"
#define PRED_ONE_TAG "pred_1sample"

#define RMSE_AVG_TAG "rmse_avg"
#define RMSE_1SAMPLE_TAG "rmse_1sample"
#define AUC_AVG_TAG "auc_avg"
#define AUC_1SAMPLE_TAG "auc_1sample"
#define SAMPLE_ITER_TAG "sample_iter"
#define BURNIN_ITER_TAG "burnin_iter"

#define LATENTS_PREFIX "latents_"
#define LINK_MATRIX_PREFIX "link_matrix_"
#define MU_PREFIX "mu_"
#define POST_MU_PREFIX "post_mu_"
#define POST_LAMBDA_PREFIX "post_lambda_"

#define CHECKPOINT_PREFIX "checkpoint_"
#define SAMPLE_PREFIX "sample_"

namespace smurff {

Step::Step(h5::File file, std::int32_t isample, bool checkpoint)
   : HDF5(file.createGroup(std::string(checkpoint ? CHECKPOINT_PREFIX : SAMPLE_PREFIX) + std::to_string(isample)))
   , m_file(file) 
   , m_isample(isample)
   , m_checkpoint(checkpoint)
{
   m_group.createAttribute<bool>(IS_CHECKPOINT_TAG, m_checkpoint);
   m_group.createAttribute<int>(NUMBER_TAG, m_isample);
}

Step::Step(h5::File file, h5::Group group)
   : HDF5(group), m_file(file)
{
   group.getAttribute(NUMBER_TAG).read(m_isample);
   group.getAttribute(IS_CHECKPOINT_TAG).read(m_checkpoint);
}

Step::~Step()
{
   if (isCheckpoint())
   {
      m_file.getAttribute(LAST_CHECKPOINT_TAG).write(getName());
   }

   m_file.flush();
}

//name methods
unsigned Step::getNModes() const
{
   unsigned nmodes;
   m_group.getAttribute(NUM_MODES_TAG).read(nmodes);
   return nmodes;
}

bool Step::hasModel(std::uint64_t index) const
{
   return hasDataSet(LATENTS_SEC_TAG, LATENTS_PREFIX + std::to_string(index));
}

std::shared_ptr<Matrix> Step::getModel(std::uint64_t index) const
{
   return getMatrix(LATENTS_SEC_TAG, LATENTS_PREFIX + std::to_string(index));
}

std::string Step::getName() const
{
   return std::string(isCheckpoint() ? CHECKPOINT_PREFIX : SAMPLE_PREFIX) + std::to_string(getIsample());
}

void Step::putModel(const std::vector<std::shared_ptr<Matrix>> &F) const
{
   m_group.createAttribute(NUM_MODES_TAG, F.size());
   for (std::uint64_t m = 0; m < F.size(); ++m)
   {
      putMatrix(LATENTS_SEC_TAG, LATENTS_PREFIX + std::to_string(m), *F[m]);
   }
}

bool Step::hasLinkMatrix(std::uint32_t mode) const
{
   return hasDataSet(LINK_MATRICES_SEC_TAG, LINK_MATRIX_PREFIX + std::to_string(mode));
}

std::shared_ptr<Matrix> Step::getLinkMatrix(std::uint32_t mode) const
{
   if (hasDataSet(LINK_MATRICES_SEC_TAG, LINK_MATRIX_PREFIX + std::to_string(mode)))
      return getMatrix(LINK_MATRICES_SEC_TAG, LINK_MATRIX_PREFIX + std::to_string(mode));

   return std::shared_ptr<Matrix>();
}

std::shared_ptr<Vector> Step::getMu(std::uint64_t index) const
{
   if (hasDataSet(LINK_MATRICES_SEC_TAG, MU_PREFIX + std::to_string(index)))
      return getVector(LINK_MATRICES_SEC_TAG, MU_PREFIX + std::to_string(index));

   return std::shared_ptr<Vector>();
}

void Step::putLinkMatrix(std::uint64_t index, const Matrix &M) const
{
   putMatrix(LINK_MATRICES_SEC_TAG, LINK_MATRIX_PREFIX + std::to_string(index), M);
}

void Step::putMu(std::uint64_t index, const Matrix &M) const
{
   putMatrix(LINK_MATRICES_SEC_TAG, MU_PREFIX + std::to_string(index), M);
}

void Step::putPostMuLambda(std::uint64_t index, const Matrix &mu, const Matrix &Lambda) const
{
   putMatrix(LATENTS_SEC_TAG, POST_MU_PREFIX + std::to_string(index), mu);
   putMatrix(LATENTS_SEC_TAG, POST_LAMBDA_PREFIX + std::to_string(index), Lambda);
}

bool Step::hasPred() const
{
   return hasDataSet(PRED_SEC_TAG, PRED_AVG_TAG);
}

void Step::putPredState(double rmse_avg, double rmse_1sample, double auc_avg, double auc_1sample,
                            int sample_iter, int burnin_iter) const
{
   auto pred_group = m_group.getGroup(PRED_SEC_TAG);
   pred_group.createAttribute<double>(RMSE_AVG_TAG, rmse_avg);
   pred_group.createAttribute<double>(RMSE_1SAMPLE_TAG, rmse_1sample);
   pred_group.createAttribute<double>(AUC_AVG_TAG, auc_avg);
   pred_group.createAttribute<double>(AUC_1SAMPLE_TAG, auc_1sample);
   pred_group.createAttribute<int>(SAMPLE_ITER_TAG, sample_iter);
   pred_group.createAttribute<int>(BURNIN_ITER_TAG, burnin_iter);
}

void Step::getPredState(
   double &rmse_avg, double &rmse_1sample, double &auc_avg, double &auc_1sample, int &sample_iter, int &burnin_iter) const
{
   auto pred_group = m_group.getGroup(PRED_SEC_TAG);
   pred_group.getAttribute(RMSE_AVG_TAG).read(rmse_avg);
   pred_group.getAttribute(RMSE_1SAMPLE_TAG).read(rmse_1sample);
   pred_group.getAttribute(AUC_AVG_TAG).read(auc_avg);
   pred_group.getAttribute(AUC_1SAMPLE_TAG).read(auc_1sample);
   pred_group.getAttribute(SAMPLE_ITER_TAG).read(sample_iter);
   pred_group.getAttribute(BURNIN_ITER_TAG).read(burnin_iter);

}

void Step::putPredAvgVar(const SparseMatrix &avg, const SparseMatrix &var, const SparseMatrix &one_sample) const
{
   putSparseMatrix(PRED_SEC_TAG, PRED_AVG_TAG, avg);
   putSparseMatrix(PRED_SEC_TAG, PRED_VAR_TAG, var);
   putSparseMatrix(PRED_SEC_TAG, PRED_ONE_TAG, one_sample);
}


std::shared_ptr<Matrix> Step::getPredAvg() const
{
   return getMatrix(PRED_SEC_TAG, PRED_AVG_TAG);
}

std::shared_ptr<Matrix> Step::getPredVar() const
{
   return getMatrix(PRED_SEC_TAG, PRED_VAR_TAG);
}

//save methods

void Step::save(
         std::shared_ptr<const Model> model,
         std::shared_ptr<const Result> pred,
   const std::vector<std::shared_ptr<ILatentPrior> >& priors
   ) const
{
   model->save(shared_from_this());
   pred->save(shared_from_this());
   for (auto &p : priors) p->save(shared_from_this());
}

//restore methods

//-- used in PredictSession
std::shared_ptr<Model> Step::restoreModel(int skip_mode) const
{
    auto model = std::make_shared<Model>();
    model->restore(shared_from_this(), skip_mode);
    return model;
}

void Step::restore(std::shared_ptr<Model> model, std::shared_ptr<Result> pred, std::vector<std::shared_ptr<ILatentPrior> >& priors) const
{
   model->restore(shared_from_this());
   pred->restore(shared_from_this());
   for (auto &p : priors) p->restore(shared_from_this());
}

//getters

std::int32_t Step::getIsample() const
{
   return m_isample;
}

bool Step::isCheckpoint() const
{
   return m_checkpoint;
}

} // end namespace