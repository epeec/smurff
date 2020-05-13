
#include "MacauPrior.h"

#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/Utils/counters.h>

#include <ios>

//#define COMPARE_CPU_GPU

namespace smurff
{

    namespace mu = smurff::matrix_utils;

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

        af::setDevice(0);

        Matrix tmp(num_feat(), num_feat());
        Features->At_mul_A(tmp);
        FtF = matrix_utils::to_af(tmp);

        Uhat.resize(num_item(), num_latent());
        Uhat.setZero();
        Uhat_lcl = mu::to_af(Uhat);

        BtB.resize(num_latent(), num_latent());
        BtB.setZero();

    }

    void MacauPrior::update_prior()
    {
        COUNTER("update_prior");

        af::setDevice(0);

        U_lcl = mu::to_af(U());

        // sampling Hyper Params
        {
            COUNTER("sample hyper");
            // uses: U, Uhat
            // writes: mu and Lambda
            // complexity: num_latent x num_items
            auto Udelta_lcl = U_lcl - Uhat_lcl;
            auto N = Udelta_lcl.dims(1);
            auto NS_lcl = af::matmulNT(Udelta_lcl, Udelta_lcl);
            auto NU_lcl = af::sum(Udelta_lcl, 1);

            Matrix NS, NU;
            mu::to_eigen(NS_lcl, NS);
            mu::to_eigen(NU_lcl, NU);
            std::tie(mu(), Lambda) = CondNormalWishart(N, NS, NU, mu0, b0, WI + beta_precision * BtB, df + num_feat());
        }

        // uses: U, F
        // writes: Ft_y
        // complexity: num_latent x num_feat x num_item
        compute_Ft_y();
        sample_beta();

        if (enable_beta_precision_sampling)
        {
            // uses: BtB 
            COUNTER("sample_beta_precision");
            beta_precision = sample_beta_precision(BtB, Lambda, beta_precision_nu0, beta_precision_mu0, num_feat());
        }

        {
            COUNTER("compute_uhat");
            // Uhat = beta * F
            // uses: beta, F
            // output: Uhat
            // complexity: num_feat x num_latent x num_item
            Uhat_lcl = af::matmul(Features->arr_t(), beta).T(); 
            matrix_utils::to_eigen(Uhat_lcl, Uhat);
        }
    }

    void MacauPrior::sample_beta()
    {
        COUNTER("sample_beta");
        // uses: FtF, Ft_y,
        // writes: beta
        // complexity: num_feat^3

        // update diagonal of FtF with new beta_precision
        af::array diag_vec = af::constant(beta_precision, num_feat());
        af::array diag_mat = af::diag(diag_vec, 0, false);
        auto FtF_beta = FtF + diag_mat;

        beta = af::solve(FtF_beta, Ft_y.T());

        // complexity: num_feat x num_feat x num_latent
        matrix_utils::to_eigen(af::matmulTN(beta, beta), BtB);
    }

    const Vector MacauPrior::fullMu(int n) const
    {
        return mu() + Uhat.row(n);
    }

    void MacauPrior::compute_Ft_y()
    {
        COUNTER("compute Ft_y");
        // Ft_y = (U .- mu + Normal(0, Lambda^-1)) * F + std::sqrt(beta_precision) * Normal(0, Lambda^-1)
        // Ft_y is [ num_latent x num_feat ] matrix

        af::array h1 = U_lcl + af_MvNormal(Lambda, num_item()) - af::tile(matrix_utils::to_af(mu()), 1, num_item());
        af::array Ft_y1 = af::matmulNT(Features->arr(), h1).T();
        af::array h2 = af_MvNormal(Lambda, num_feat());
        Ft_y = Ft_y1 + h2 * std::sqrt(beta_precision);
    }

    void MacauPrior::addSideInfo(const std::shared_ptr<ISideInfo> &side, double bp, double to, bool di, bool sa, bool th)
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

    std::ostream &MacauPrior::info(std::ostream &os, std::string indent)
    {
        NormalPrior::info(os, indent);
        os << indent << " SideInfo: ";
        Features->print(os);
        os << indent << " Method: ";
        if (use_FtF)
        {
            os << "Cholesky Decomposition";
            double needs_gb = (double)num_feat() / 1024. * (double)num_feat() / 1024. / 1024.;
            if (needs_gb > 1.0)
                os << " (needing " << needs_gb << " GB of memory)";
            os << std::endl;
        }
        else
        {
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
        os << indent << "mu           = " << mu() << std::endl;
        os << indent << "Uhat mean    = " << Uhat.colwise().mean() << std::endl;
        if (!use_FtF) os << indent << "blockcg iter = " << blockcg_iter << std::endl;
        os << indent << "FtF          = " << af::norm(FtF) << std::endl;
        os << indent << "Beta         = " << af::norm(beta) << std::endl;
        os << indent << "beta_precision  = " << beta_precision << std::endl;
        return os;
    }

    std::pair<double, double> MacauPrior::posterior_beta_precision(const Matrix &BtB, Matrix &Lambda_u, double nu, double mu, int N)
    {
        double nux = nu + N * BtB.cols();
        double mux = mu * nux / (nu + mu * (BtB.selfadjointView<Eigen::Lower>() * Lambda_u).trace());
        double b = nux / 2;
        double c = 2 * mux / nux;
        return std::make_pair(b, c);
    }

    double MacauPrior::sample_beta_precision(const Matrix &BtB, Matrix &Lambda_u, double nu, double mu, int N)
    {
        auto gamma_post = posterior_beta_precision(BtB, Lambda_u, nu, mu, N);
        return rand_gamma(gamma_post.first, gamma_post.second);
    }
} // end namespace smurff
