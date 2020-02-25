#include "DataCreatorBase.h"

#include <SmurffCpp/Utils/MatrixUtils.h>

//matrix classes
#include <SmurffCpp/DataMatrices/SparseMatrixData.h>
#include <SmurffCpp/DataMatrices/ScarceMatrixData.h>
#include <SmurffCpp/DataMatrices/DenseMatrixData.h>

//tensor classes
#include <SmurffCpp/DataTensors/TensorData.h>

//noise classes
#include <SmurffCpp/Configs/NoiseConfig.h>
#include <SmurffCpp/Noises/NoiseFactory.h>

namespace smurff {

std::shared_ptr<Data> DataCreatorBase::create(const DataConfig &dc) const
{
   std::shared_ptr<INoiseModel> noise = NoiseFactory::create_noise_model(dc.getNoiseConfig());
   std::shared_ptr<Data> ret;
   
   if (dc.isMatrix())
   {
      if (dc.isDense())
        ret = std::make_shared<DenseMatrixData>(dc.getDenseMatrixData());
      else if (!dc.isScarce())
        ret = std::make_shared<SparseMatrixData>(dc.getSparseMatrixData());
      else
        ret = std::make_shared<ScarceMatrixData>(dc.getSparseMatrixData());
   }
   else
   {
      if (dc.isDense())
         ret = std::make_shared<TensorData>(dc.getDenseTensorData());
      else if (!dc.isScarce())
        ret = std::make_shared<TensorData>(dc.getSparseTensorData()); // FIXME
      else
        ret = std::make_shared<TensorData>(dc.getSparseTensorData()); // FIXME
   }

   ret->setNoiseModel(noise);
   return ret;
}

} // end namespace smurff
