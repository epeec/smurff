#include "SparseMatrixData.h"

namespace smurff {

SparseMatrixData::SparseMatrixData(SparseMatrix Y)
   : FullMatrixData<SparseMatrix>(Y)
{
   this->name = "SparseMatrixData [fully known]";
}

void SparseMatrixData::getMuLambda(const SubModel& model, uint32_t mode, int d, Vector& rr, Matrix& MM) const
{
    const auto& Y = this->Y(mode);
    auto Vf = *model.CVbegin(mode);
    auto &ns = noise();

    for (SparseMatrix::InnerIterator it(Y, d); it; ++it) 
    {
        const auto &row = Vf.row(it.col());
        auto p = pos(mode, d, it.col());
        double noisy_val = ns.sample(model, p, it.value());
        rr.noalias() += row * noisy_val; // rr = rr + (V[m] * y[d]) * alpha
    }

    MM.noalias() += ns.getAlpha() * VV[mode]; // MM = MM + VV[m]
}

double SparseMatrixData::train_rmse(const SubModel& model) const
{
   return std::sqrt(sumsq(model) / this->size());
}

double SparseMatrixData::var_total() const
{
   const double cwise_mean = this->sum() / this->size();
   const double cwise_mean_squared = std::pow(cwise_mean, 2);
   double se = 0.0;

   #pragma omp parallel for schedule(guided) reduction(+:se)
   for(int c = 0; c < Y().outerSize(); ++c)
   {
      int r = 0;
      for (SparseMatrix::InnerIterator it(Y(), c); it; ++it)
      {
         se += (r - it.row()) * cwise_mean_squared; // handle implicit zeroes
         se += std::pow(it.value() - cwise_mean, 2);
         r = it.row() + 1;
      }

      se += (r - Y().rows()) * cwise_mean_squared; // handle implicit zeroes
   }

   double var = se / this->size();
   if (var <= 0.0 || std::isnan(var))
   {
      // if var cannot be computed using 1.0
      var = 1.0;
   }

   return var;
}

double SparseMatrixData::sumsq(const SubModel& model) const
{
   double sumsq = 0.0;
   
   THROWERROR_ASSERT(Y().IsRowMajor);
   #pragma omp parallel for schedule(guided) reduction(+:sumsq)
   for(int r = 0; r < Y().outerSize(); ++r) // rows
   {
      int c = 0;
      for (SparseMatrix::InnerIterator it(Y(), r); it; ++it) // cols
      {
         for(; c < it.col(); c++) //handle implicit zeroes
            sumsq += std::pow(model.predict({r, c}), 2);

         // actual non-zero
         sumsq += std::pow(model.predict({r, c}) - it.value(), 2);
         c++;
      }

      for(; c < Y().cols(); c++) //handle implicit zeroes
         sumsq += std::pow(model.predict({r, c}), 2);
   }

   return sumsq;
}
} // end namespace smurff
