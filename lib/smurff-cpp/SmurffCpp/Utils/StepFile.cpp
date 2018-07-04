#include <SmurffCpp/Utils/StepFile.h>

#include <iostream>

#include <SmurffCpp/Model.h>
#include <SmurffCpp/result.h>
#include <SmurffCpp/Priors/ILatentPrior.h>

#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/Utils/StringUtils.h>
#include <SmurffCpp/IO/GenericIO.h>
#include <SmurffCpp/IO/MatrixIO.h>

#define NONE_TAG "none"

#define STEP_SAMPLE_PREFIX "sample-"
#define STEP_CHECKPOINT_PREFIX "checkpoint-"
#define STEP_INI_SUFFIX "-step.ini"

#define LATENTS_PREFIX "latents_"
#define LINK_MATRIX_PREFIX "link_matrix_"

#define GLOBAL_SEC_TAG "global"
#define LATENTS_SEC_TAG "latents"
#define PRED_SEC_TAG "predictions"
#define LINK_MATRICES_SEC_TAG "link_matrices"

#define IS_CHECKPOINT_TAG "is_checkpoint"
#define NUMBER_TAG "number"
#define NUM_MODES_TAG "num_modes"
#define PRED_TAG "pred"
#define PRED_STATE_TAG "pred_state"

using namespace smurff;

StepFile::StepFile(std::int32_t isample, std::string prefix, std::string extension, bool create, bool checkpoint)
   : m_isample(isample), m_prefix(prefix), m_extension(extension), m_checkpoint(checkpoint)
{
   if (create)
   {
      m_iniReader = std::make_shared<INIFile>();
      m_iniReader->create(getStepFileName());
   }
   else
   {
      //load all entries in ini file to be able to go through step file internals
      m_iniReader = std::make_shared<INIFile>();
      m_iniReader->open(getStepFileName());
   }
}

StepFile::StepFile(const std::string& path, std::string prefix, std::string extension)
   : m_prefix(prefix), m_extension(extension)
{
   //load all entries in ini file to be able to go through step file internals
   m_iniReader = std::make_shared<INIFile>();
   m_iniReader->open(path);

   m_checkpoint = std::stoi(getIniValueBase(GLOBAL_SEC_TAG, IS_CHECKPOINT_TAG));
   m_isample = std::stoi(getIniValueBase(GLOBAL_SEC_TAG, NUMBER_TAG));
}

//name methods
std::pair<bool, std::string> StepFile::tryGetIniValueFullPath(const std::string &section, const std::string &tag) const
{
   auto pair = tryGetIniValueBase(section, tag);

   if (pair.first && !startsWith(pair.second, m_prefix))
      pair.second = m_prefix + pair.second;

   return pair;
}

std::string StepFile::getStepFileName() const
{
   std::string prefix = getStepPrefix();
   return prefix + STEP_INI_SUFFIX;
}

bool StepFile::isBinary() const
{
    if (m_extension == ".ddm")
    {
        return true;
    }
    else
    {
        THROWERROR_ASSERT_MSG(m_extension == ".csv", "Invalid save_extension: " + m_extension);
    }
    return false;
}

std::string StepFile::getStepPrefix() const
{
   std::string prefix = m_checkpoint ? STEP_CHECKPOINT_PREFIX : STEP_SAMPLE_PREFIX;
   return m_prefix + prefix + std::to_string(m_isample);
}

std::string StepFile::getModelFileName(std::uint64_t index) const
{
   auto modelIt = tryGetIniValueFullPath(LATENTS_SEC_TAG, LATENTS_PREFIX + std::to_string(index));
   if (modelIt.first)
      return modelIt.second; 

   std::string prefix = getStepPrefix();
   return prefix + "-U" + std::to_string(index) + "-latents" + m_extension;
}

std::string StepFile::getLinkMatrixFileName(std::uint32_t mode) const
{
   auto linkMatrixIt = tryGetIniValueFullPath(LINK_MATRICES_SEC_TAG, LINK_MATRIX_PREFIX + std::to_string(mode));
   if (linkMatrixIt.first)
      return linkMatrixIt.second; 

   std::string prefix = getStepPrefix();
   return prefix + "-F" + std::to_string(mode) + "-link" + m_extension;
}

std::string StepFile::getPredFileName() const
{
   auto predIt = tryGetIniValueFullPath(PRED_SEC_TAG, PRED_TAG);
   if (predIt.first)
      return predIt.second; 

   std::string prefix = getStepPrefix();
   std::string extension = isBinary() ? ".bin" : ".csv";
   return prefix + "-predictions" + extension;
}

std::string StepFile::getPredStateFileName() const
{
   auto predStateIt = tryGetIniValueFullPath(PRED_SEC_TAG, PRED_STATE_TAG);
   if (predStateIt.first)
      return predStateIt.second; 

   std::string prefix = getStepPrefix();
   return prefix + "-predictions-state.ini";
}

//save methods

void StepFile::saveModel(std::shared_ptr<const Model> model) const
{
   model->save(shared_from_this());

   //save models
   appendToStepFile(LATENTS_SEC_TAG, NUM_MODES_TAG, std::to_string(model->nmodes()));

   for (std::uint64_t mIndex = 0; mIndex < model->nmodes(); mIndex++)
   {
      std::string path = getModelFileName(mIndex);
      appendToStepFile(LATENTS_SEC_TAG, LATENTS_PREFIX + std::to_string(mIndex), path);
   }
}

void StepFile::savePred(std::shared_ptr<const Result> m_pred) const
{
   if (m_pred->isEmpty())
      return;

   m_pred->save(shared_from_this());

   //save predictions

   appendToStepFile(PRED_SEC_TAG, PRED_TAG, getPredFileName());
   appendToStepFile(PRED_SEC_TAG, PRED_STATE_TAG, getPredStateFileName());
}

void StepFile::savePriors(const std::vector<std::shared_ptr<ILatentPrior> >& priors) const
{
   std::uint64_t pIndex = 0;
   for (auto &p : priors)
   {
      if (p->save(shared_from_this()))
      {
          std::string priorPath = getLinkMatrixFileName(priors.at(pIndex)->getMode());
          appendToStepFile(LINK_MATRICES_SEC_TAG, LINK_MATRIX_PREFIX + std::to_string(pIndex), priorPath);
      }

      pIndex++;
   }

   //save priors
   for (std::uint64_t pIndex = 0; pIndex < priors.size(); pIndex++)
   {
   }
}

void StepFile::save(std::shared_ptr<const Model> model, std::shared_ptr<const Result> pred, const std::vector<std::shared_ptr<ILatentPrior> >& priors) const
{
    appendToStepFile(GLOBAL_SEC_TAG, IS_CHECKPOINT_TAG, std::to_string(m_checkpoint));
    appendToStepFile(GLOBAL_SEC_TAG, NUMBER_TAG, std::to_string(m_isample));
    appendToStepFile(GLOBAL_SEC_TAG, NUM_MODES_TAG, std::to_string(model->nmodes()));

    saveModel(model);
    savePred(pred);
    savePriors(priors);
}

//restore methods

void StepFile::restoreModel(std::shared_ptr<Model> model) const
{
   //it is enough to check presence of num tag
   if (!hasIniValueBase(LATENTS_SEC_TAG, NUM_MODES_TAG))
      return;

   model->restore(shared_from_this());

   int nmodes = model->nmodes();
   for(int i=0; i<nmodes; ++i)
   {
       auto linkMatrixIt = tryGetIniValueFullPath(LINK_MATRICES_SEC_TAG, LINK_MATRIX_PREFIX + std::to_string(i));
       if (!linkMatrixIt.first)
           continue;

       std::string path = linkMatrixIt.second;
       THROWERROR_FILE_NOT_EXIST(path);
       auto beta = std::make_shared<Eigen::MatrixXd>();
       matrix_io::eigen::read_matrix(path, *beta); 

       model->setLinkMatrix(i, beta);
   }
}

  //-- used in PredictSession
std::shared_ptr<Model> StepFile::restoreModel() const
{
    auto model = std::make_shared<Model>();
    restoreModel(model);
    return model;
}

void StepFile::restorePred(std::shared_ptr<Result> m_pred) const
{
   if (!hasIniValueBase(PRED_SEC_TAG, PRED_TAG))
      return;

   if (!hasIniValueBase(PRED_SEC_TAG, PRED_STATE_TAG))
      return;

   m_pred->restore(shared_from_this());
}

void StepFile::restorePriors(std::vector<std::shared_ptr<ILatentPrior> >& priors) const
{
   for (auto &p : priors)
   {
      p->restore(shared_from_this());
   }
}

void StepFile::restore(std::shared_ptr<Model> model, std::shared_ptr<Result> pred, std::vector<std::shared_ptr<ILatentPrior> >& priors) const
{
   restoreModel(model);
   restorePred(pred);
   restorePriors(priors);
}

//remove methods

void StepFile::removeModel() const
{
   std::uint64_t index = 0;
   while (true)
   {
      std::string path = getModelFileName(index++);
      if (!generic_io::file_exists(path))
         break;

      std::remove(path.c_str());
   }

   std::int32_t nModels = getNModes();
   for(std::int32_t i = 0; i < nModels; i++)
      removeFromStepFile(LATENTS_SEC_TAG, LATENTS_PREFIX + std::to_string(i));
}

void StepFile::removePred() const
{
   std::remove(getPredFileName().c_str());
   removeFromStepFile(PRED_SEC_TAG, PRED_TAG);

   std::remove(getPredStateFileName().c_str());
   removeFromStepFile(PRED_SEC_TAG, PRED_STATE_TAG);
}

void StepFile::removePriors() const
{
   std::uint32_t mode = 0;
   while (true)
   {
      std::string path = getLinkMatrixFileName(mode++);
      if (path == NONE_TAG)
         break;

      std::remove(path.c_str());
   }

   for (std::int32_t i = 0; i < getNModes(); i++)
      removeFromStepFile(LINK_MATRICES_SEC_TAG, LINK_MATRIX_PREFIX + std::to_string(i));
}

void StepFile::remove(bool model, bool pred, bool priors) const
{
   if (m_iniReader->empty()) 
       return;

   //remove all model files
   if(model)
      removeModel();

   //remove all pred files
   if(pred)
      removePred();

   //remove all prior files
   if(priors)
      removePriors();

   //remove step file itself
   std::remove(getStepFileName().c_str());

   THROWERROR_ASSERT_MSG(m_iniReader->empty(), "Unexpected data in step file");

   //nullify reader
   m_iniReader = std::shared_ptr<INIFile>();
}

//getters

std::int32_t StepFile::getIsample() const
{
   return m_isample;
}

bool StepFile::isCheckpoint() const
{
   return m_checkpoint;
}

std::int32_t StepFile::getNModes() const
{
   return std::stoi(getIniValueBase(LATENTS_SEC_TAG, NUM_MODES_TAG));
}

//ini methods

std::string StepFile::getIniValueBase(const std::string& section, const std::string& tag) const
{
   THROWERROR_ASSERT_MSG(m_iniReader, "Step ini file is not loaded");

   return m_iniReader->get(section, tag);
}

bool StepFile::hasIniValueBase(const std::string& section, const std::string& tag) const
{
   return tryGetIniValueBase(section, tag).first;
}

std::pair<bool, std::string> StepFile::tryGetIniValueBase(const std::string& section, const std::string& tag) const
{
   if (m_iniReader)
      return m_iniReader->tryGet(section, tag);

   return std::make_pair(false, std::string());
}

void StepFile::appendToStepFile(std::string section, std::string tag, std::string value) const
{
   if (m_cur_section != section) {
      m_iniReader->startSection(section);
      m_cur_section = section;
   }

   value = stripPrefix(value, m_prefix);

   m_iniReader->appendItem(section, tag, value);
   
   flushLast();
}

void StepFile::appendCommentToStepFile(std::string comment) const
{
   m_iniReader->appendComment(comment);
}

void StepFile::removeFromStepFile(std::string section, std::string tag) const
{
   m_iniReader->removeItem(std::string(), tag);
}


void StepFile::flushLast() const
{
   m_iniReader->flush();
}
