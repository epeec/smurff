#include <memory>

#include "Data.h"
#include <SmurffCpp/Utils/Error.h>

//matrix classes
#include <SmurffCpp/DataMatrices/SparseMatrixData.h>
#include <SmurffCpp/DataMatrices/ScarceMatrixData.h>
#include <SmurffCpp/DataMatrices/DenseMatrixData.h>
#include <SmurffCpp/DataMatrices/MatricesData.h>

//tensor classes
#include <SmurffCpp/DataTensors/TensorData.h>

//noise classes
#include <SmurffCpp/Configs/NoiseConfig.h>
#include <SmurffCpp/Noises/NoiseFactory.h>

#include <SmurffCpp/Configs/DataConfig.h>

namespace smurff {

Data::Data()
{
}

void Data::init()
{
    init_pre();

    init_post();
}

void Data::init_post()
{
   noise().init(this);
}

void Data::update(const SubModel& model)
{
   noise().update(model);
}

//#### dimension functions ####

std::uint64_t Data::size() const
{
   return dim().dot();
}

int Data::dim(int m) const
{
   return dim().at(m);
}

//#### view functions ####

int Data::nview(int mode) const
{
   return 1;
}

int Data::view(int mode, int pos) const
{
   return 0;
}

int Data::view_size(int m, int v) const
{
    return this->dim(m);
}

//#### noise, precision, mean functions ####

INoiseModel &Data::noise() const
{
   THROWERROR_ASSERT(noise_ptr != 0);
   
   return *noise_ptr;
}

void Data::setNoiseModel(std::unique_ptr<INoiseModel> &&nm)
{
   noise_ptr = std::move(nm);
}

//#### info functions ####

std::ostream& Data::info(std::ostream& os, std::string indent)
{
   double cwise_mean = this->sum() / (this->size() - this->nna());

   os << indent << "Type: " << name << std::endl;
   os << indent << "Component-wise mean: " << cwise_mean << std::endl;
   os << indent << "Component-wise variance: " << var_total() << std::endl;
   os << indent << "Noise: ";
   noise().info(os, "");
   return os;
}

std::ostream& Data::status(std::ostream& os, std::string indent) const
{
   os << indent << noise().getStatus() << std::endl;
   return os;
}

std::shared_ptr<Data> Data::create(const std::vector<DataConfig> &dcs)
{
   //create single matrix or tensor -- only train
   if (dcs.size() == 1) return Data::create(dcs.at(0));

   //multiple matrices
   THROWERROR_ASSERT_MSG(dcs.at(0).isMatrix(), "Tensor config does not support aux data");

   NoiseConfig ncfg(NoiseTypes::unused);
   auto data_ptr = std::make_shared<MatricesData>();
   data_ptr->setNoiseModel(NoiseFactory::create_noise_model(ncfg));

   for (auto &m : dcs)
      data_ptr->add(m.getPos(), Data::create(m));

   return data_ptr;
}

std::shared_ptr<Data> Data::create(const DataConfig& dc) 
{
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

   ret->setNoiseModel(NoiseFactory::create_noise_model(dc.getNoiseConfig()));
   return ret;
}

} // end namespace smurff
