
#include "MacauPrior.h"

#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/Utils/counters.h>

#include <ios>

#define COMPARE_CPU_GPU

namespace smurff {

MacauPrior::MacauPrior(TrainSession &trainSession, uint32_t mode)
    : NormalPrior(trainSession, mode, "MacauPrior"), blockcg_iter(-1)
{
    beta_precision = SideInfoConfig::BETA_PRECISION_DEFAULT_VALUE;
    tol = SideInfoConfig::TOL_DEFAULT_VALUE;

    enable_beta_precision_sampling = Config::ENABLE_BETA_PRECISION_SAMPLING_DEFAULT_VALUE;
}

MacauPrior::~MacauPrior()
{
}

void MacauPrior::init()
{
   NormalPrior::init();

   THROWERROR_ASSERT_MSG(Features->rows() == num_item(), "Number of rows in train must be equal to number of rows in features");

   if (use_FtF)
   {
       std::uint64_t dim = num_feat();
       FtF_plus_precision.resize(dim, dim);
       Features->At_mul_A(FtF_plus_precision);
       FtF_plus_precision.diagonal().array() += beta_precision;

       gpu_FtF = af::array(num_feat(), num_feat(), FtF_plus_precision.data());
   }

   Uhat.resize(num_item(), num_latent());
   Uhat.setZero();

   beta().resize(num_feat(), num_latent());
   beta().setZero();

   BtB = beta().transpose() * beta();
}

void MacauPrior::update_prior()
{
    COUNTER("update_prior");

    // sampling Hyper Params
    {
        COUNTER("sample hyper mu/Lambda");
        // uses: U, Uhat
        // writes: mu and Lambda
        // complexity: num_latent x num_items
        std::tie(mu(), Lambda) = CondNormalWishart(U() - Uhat, mu0, b0, WI + beta_precision * BtB, df + num_feat());
    }

    // uses: U, F
    // writes: Ft_y
    // complexity: num_latent x num_feat x num_item
    compute_Ft_y(Ft_y);

    sample_beta();

    prev_beta_precision = beta_precision;
    if (enable_beta_precision_sampling)
    {
        // uses: BtB
        // writes: FtF
        COUNTER("sample_beta_precision");
        beta_precision = sample_beta_precision(BtB, Lambda, beta_precision_nu0, beta_precision_mu0, beta().rows());
        #ifdef COMPARE_CPU_GPU
        FtF_plus_precision.diagonal().array() += beta_precision - prev_beta_precision;
        #endif
    }

    {
        COUNTER("compute_uhat");
        // Uhat = beta * F
        // uses: beta, F
        // output: Uhat
        // complexity: num_feat x num_latent x num_item
        Features->compute_uhat(Uhat, beta());
    }
}

void MacauPrior::sample_beta()
{
    COUNTER("sample_beta");
    if (use_FtF)
    {
            // uses: FtF, Ft_y,
            // writes: beta()
            // complexity: num_feat^3

            // update diagonal of FtD with new beta_precision
            af::array gpu_Ft_y = af::array(Ft_y.rows(), Ft_y.cols(), Ft_y.data());
            float_type update_beta = beta_precision - prev_beta_precision;
            af::array new_diag = af::diag(af::constant(update_beta, num_feat()), 0, false);
            gpu_FtF += new_diag;

            // update llt(FtF)
            af::array gpu_FtF_lu, gpu_FtF_pivot;
            af::lu(gpu_FtF_lu, gpu_FtF_pivot, gpu_FtF);
            af::array gpu_beta = af::solveLU(gpu_FtF_lu, gpu_FtF_pivot, gpu_Ft_y);

            //copy back
            gpu_beta.host(beta().data());

        #ifdef COMPARE_CPU_GPU
            // for verification
            SHOW(beta().norm());
            Matrix cpu_beta = FtF_plus_precision.llt().solve(Ft_y);
            SHOW(cpu_beta.norm());
            Matrix cpu_diff = (FtF_plus_precision * cpu_beta) - Ft_y;
            SHOW(cpu_diff.norm());
            Matrix gpu_diff = (FtF_plus_precision * beta()) - Ft_y;
            SHOW(gpu_diff.norm());
        #endif
    }

    else
    {
        // uses: Features, beta_precision, Ft_y,
        // writes: beta
        // complexity: num_feat x num_feat x num_iter
        blockcg_iter = Features->solve_blockcg(beta(), beta_precision, Ft_y, tol, 32, 8, throw_on_cholesky_error);
    }
    // complexity: num_feat x num_feat x num_latent
    BtB = beta().transpose() * beta();
}

const Vector MacauPrior::fullMu(int n) const
{
   return mu() + Uhat.row(n);
}

void MacauPrior::compute_Ft_y(Matrix& Ft_y)
{
    COUNTER("compute Ft_y");
   // Ft_y = (U .- mu + Normal(0, Lambda^-1)) * F + std::sqrt(beta_precision) * Normal(0, Lambda^-1)
   // Ft_y is [ num_latent x num_feat ] matrix

   //HyperU: num_latent x num_item
   HyperU = (U() + MvNormal(Lambda, num_item())).rowwise() - mu();
   Ft_y = Features->A_mul_B(HyperU); // num_latent x num_feat

   //--  add beta_precision 
   HyperU2 = MvNormal(Lambda, num_feat()); // num_latent x num_feat
   Ft_y += std::sqrt(beta_precision) * HyperU2;
}

void MacauPrior::addSideInfo(const std::shared_ptr<ISideInfo>& side, double bp, double to, bool di, bool sa, bool th)
{
    Features = side;
    beta_precision = bp;
    tol = to;
    use_FtF = di;
    enable_beta_precision_sampling = sa;
    throw_on_cholesky_error = th;

    // Hyper-prior for beta_precision (mean 1.0, var of 1e+3):
    beta_precision_mu0 = 1.0;
    beta_precision_nu0 = 1e-3;
}

std::ostream& MacauPrior::info(std::ostream &os, std::string indent)
{
   NormalPrior::info(os, indent);
   os << indent << " SideInfo: ";
   Features->print(os);
   os << indent << " Method: ";
   if (use_FtF)
   {
      os << "Cholesky Decomposition";
      double needs_gb = (double)num_feat() / 1024. * (double)num_feat() / 1024. / 1024.;
      if (needs_gb > 1.0) os << " (needing " << needs_gb << " GB of memory)";
      os << std::endl;
   } else {
      os << "CG Solver with tolerance: " << std::scientific << tol << std::fixed << std::endl;
   }
   os << indent << " BetaPrecision: ";
   if (enable_beta_precision_sampling)
   {
       os << "sampled around ";
   }
   else
   {
       os << "fixed at ";
   }
   os << beta_precision << std::endl;
   return os;
}

std::ostream &MacauPrior::status(std::ostream &os, std::string indent) const
{
   os << indent << m_name << ": " << std::endl;
   indent += "  ";
   os << indent << "mu           = " <<  mu() << std::endl;
   os << indent << "Uhat mean    = " <<  Uhat.colwise().mean() << std::endl;
   os << indent << "blockcg iter = " << blockcg_iter << std::endl;
   os << indent << "FtF_plus_prec= " << FtF_plus_precision.norm() << std::endl;
   os << indent << "HyperU       = " << HyperU.norm() << std::endl;
   os << indent << "HyperU2      = " << HyperU2.norm() << std::endl;
   os << indent << "Beta         = " << beta().norm() << std::endl;
   os << indent << "beta_precision  = " << beta_precision << std::endl;
   os << indent << "Ft_y         = " << Ft_y.norm() << std::endl;
   return os;
}

std::pair<double, double> MacauPrior::posterior_beta_precision(const Matrix & BtB, Matrix & Lambda_u, double nu, double mu, int N)
{
   double nux = nu + N * BtB.cols();
   double mux = mu * nux / (nu + mu * (BtB.selfadjointView<Eigen::Lower>() * Lambda_u).trace());
   double b = nux / 2;
   double c = 2 * mux / nux;
   return std::make_pair(b, c);
}

double MacauPrior::sample_beta_precision(const Matrix & BtB, Matrix & Lambda_u, double nu, double mu, int N)
{
   auto gamma_post = posterior_beta_precision(BtB, Lambda_u, nu, mu, N);
   return rand_gamma(gamma_post.first, gamma_post.second);
}
} // end namespace smurff
