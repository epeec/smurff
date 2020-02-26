#include <iostream>

#include <SmurffCpp/Utils/Step.h>

#include <SmurffCpp/Model.h>
#include <SmurffCpp/result.h>
#include <SmurffCpp/Priors/ILatentPrior.h>

#include <Utils/Error.h>
#include <Utils/StringUtils.h>


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

void Step::readModel(std::uint64_t index, Matrix &m) const
{
   read(LATENTS_SEC_TAG, LATENTS_PREFIX + std::to_string(index), m);
}

std::string Step::getName() const
{
   return std::string(isCheckpoint() ? CHECKPOINT_PREFIX : SAMPLE_PREFIX) + std::to_string(getIsample());
}

void Step::putModel(const std::vector<Matrix> &F)
{
   m_group.createAttribute(NUM_MODES_TAG, F.size());
   for (std::uint64_t m = 0; m < F.size(); ++m)
   {
      write(LATENTS_SEC_TAG, LATENTS_PREFIX + std::to_string(m), F[m]);
   }
}

void Step::readLinkMatrix(std::uint32_t mode, Matrix &m) const
{
      read(LINK_MATRICES_SEC_TAG, LINK_MATRIX_PREFIX + std::to_string(mode), m);
}

void Step::readMu(std::uint64_t index, Vector &v) const
{
   read(LINK_MATRICES_SEC_TAG, MU_PREFIX + std::to_string(index), v);
}

void Step::putLinkMatrix(std::uint64_t index, const Matrix &M)
{
   write(LINK_MATRICES_SEC_TAG, LINK_MATRIX_PREFIX + std::to_string(index), M);
}

void Step::putMu(std::uint64_t index, const Matrix &M) 
{
   write(LINK_MATRICES_SEC_TAG, MU_PREFIX + std::to_string(index), M);
}

void Step::putPostMuLambda(std::uint64_t index, const Matrix &mu, const Matrix &Lambda)
{
   write(LATENTS_SEC_TAG, POST_MU_PREFIX + std::to_string(index), mu);
   write(LATENTS_SEC_TAG, POST_LAMBDA_PREFIX + std::to_string(index), Lambda);
}

bool Step::hasPred() const
{
   return hasDataSet(PRED_SEC_TAG, PRED_AVG_TAG);
}

void Step::putPredState(double rmse_avg, double rmse_1sample, double auc_avg, double auc_1sample,
                            int sample_iter, int burnin_iter)
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

void Step::putPredAvgVar(const SparseMatrix &avg, const SparseMatrix &var, const SparseMatrix &one_sample)
{
   write(PRED_SEC_TAG, PRED_AVG_TAG, avg);
   write(PRED_SEC_TAG, PRED_VAR_TAG, var);
   write(PRED_SEC_TAG, PRED_ONE_TAG, one_sample);
}


void Step::readPredAvg(Matrix &m) const
{
   read(PRED_SEC_TAG, PRED_AVG_TAG, m);
}

void Step::readPredVar(Matrix &m) const
{
   read(PRED_SEC_TAG, PRED_VAR_TAG, m);
}

//save methods

void Step::save(
         const Model &model,
         std::shared_ptr<const Result> pred,
   const std::vector<std::shared_ptr<ILatentPrior> >& priors
   )
{
   model.save(*this);
   pred->save(*this);
   for (auto &p : priors) p->save(*this);
}

//restore methods

//-- used in PredictSession
void Step::restore(Model &model, std::shared_ptr<Result> pred, std::vector<std::shared_ptr<ILatentPrior> >& priors) const
{
   model.restore(*this);
   pred->restore(*this);
   for (auto &p : priors) p->restore(*this);
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
