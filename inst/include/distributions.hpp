/*
  Copyright (C) 2005 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

#ifndef BOOM_DISTRIBUTIONS_HPP
#define BOOM_DISTRIBUTIONS_HPP

#include <cpputil/ThrowException.hpp>
#include <distributions/Rmath_dist.hpp>
#include <distributions/rng.hpp>
#include <LinAlg/Types.hpp>
#include <vector>
#include <uint.hpp>

namespace BOOM{
  void set_seed(unsigned int I1, unsigned int I2);
  class VectorView;
  class ConstVectorView;


   class TnSampler{
     // Implements adaptive rejection sampling for drawing from the
     // truncated normal distribution given x>a>0.
    public:
     TnSampler(double a);              // Set the truncation point.
     double draw(RNG & );              // simluate a value
     void add_point(double x);         // adds the point to the hull
     double f(double x)const;          // log of the target distribution
     double df(double x)const;         // derivative of logf at x
     double h(double x, uint k)const;  // evaluates the outer hull at x
     std::ostream & print(std::ostream & out)const;
    private:
     std::vector<double> x;
     // points that have been tried thus far, stored in ascending
     // order

     std::vector<double> logf;
     // function values corresponding to values in x

     std::vector<double> dlogf;
     // derivatives of the log target density evaluated at x

     std::vector<double> knots;
     // contains the points of intersection between the tangent lines
     // to logf at x.  First knot is x[0].  Later knots satisfy x[i-1]
     // < knots[i] < x[i].

     std::vector<double> cdf;     // cdf[i] = cdf[i-1] + the integral of
     // the hull from knots[i] to
     // knots[i+1].  cdf.back() assumes a
     // final knot at infinity

     void update_cdf();
     void refresh_knots();
     double compute_knot(uint k)const;
     typedef std::vector<double>::iterator IT;
   };

  class Tn2Sampler{
    // implements adaptive rejection sampling for drawinf from the
    // truncated standard normal distribution with 0 < lo < x < hi
    public:
    Tn2Sampler(double lo, double hi);
    double draw(RNG &);
    void add_point(double x);
    double f(double x)const;
    double df(double x)const;

    // hull is the envelope distribution on the log scale
    double hull(double x, uint k)const;
    private:
    // x contains the values of the points that have been tried.
    // initialized with lo and hi
    std::vector<double> x;

    // logf contains the values of the target density at all the
    // points in x.
    std::vector<double> logf;

    // derivatives corresponding to logf
    std::vector<double> dlogf;

    // first knot is at lo.  last is at hi.  interior knots contain
    // the point of intersection of the tangent lines to logf at the
    // points in x
    std::vector<double> knots;

    // cdf[0] = integral of first bit of hull.  cdf[i] = cdf[i-1] +
    // integral of hull part i;
    std::vector<double> cdf;

     void update_cdf();
     void refresh_knots();
     double compute_knot(uint k)const;
     typedef std::vector<double>::iterator IT;
  };

  double trun_norm(double);
  double trun_norm_mt(RNG &, double);

  // Returns the mean and the variance of the truncated normal
  // distribution, where the untruncated distribution is N(mu, sigma).
  // If positive_support is true, the region of support is from
  // cutpoint to infinity.  Otherwise the region of support is from
  // cutpoint to -infinity.
  //
  // On output, *mean and *variance contain the mean and variance of
  // the truncated distribution.
  //
  // Be aware that the standard deviation of the untruncated
  // distribution is input.  The variance of the truncated
  // distribution is output.
  void trun_norm_moments(
      double mu, double sigma, double cutpoint, bool positive_support,
      double *mean, double *variance);

  double rtrun_norm(double mu, double sigma,
                    double cutpoint, bool positive_support = true);
  double rtrun_norm_mt(RNG &, double mu, double sigma,
                       double cutpoint, bool positive_support = true);

  double dtrun_norm(double, double, double, double,
		    bool low=true, bool log=false);
  double dtrun_norm_2(double, double, double, double, double, bool log=false);

  double rtrun_norm_2(double mu, double sig, double lo, double hi);
  double rtrun_norm_2_mt(RNG &, double mu, double sig, double lo, double hi);

  double dstudent(double, double, double, double, bool log=false);
  double rstudent(double, double, double);
  double rstudent_mt(RNG & rng, double, double, double);

  double rtrun_exp_mt(RNG & rng, double lam, double lo, double hi);
  double rtrun_exp(double lam, double lo, double hi);

  double rlexp(double loglam);  // log E(lam).  loglam = log(lam)
  double rlexp_mt(RNG & rng, double loglam);  // log E(lam).  loglam = log(lam)

  // extreme value distribution with mean 'loc'
  // and variance 'scale^2 * pi^2/6'
  double dexv(double x, double loc = 0., double scale=1., bool log=false);
  double rexv_mt(RNG & rng, double loc = 0., double scale=1.);
  double rexv(double loc = 0., double scale=1.);

  // random integer uniform on lo to hi, inclusive
  int random_int(int lo, int hi);
  int random_int_mt(RNG & rng, int lo, int hi);


  // basic rmvn checks the cholesky decomposition, if there is a
  // problem it calls rmvn_robust
  Vec rmvn(const Vec &Mu, const Spd &Sigma);
  Vec rmvn_mt(RNG & rng, const Vec &Mu, const Spd &Sigma);

  // rmvn_robust computes the spectral decomposition of Sigma which
  // can be done even if there is a zero pivot that would prevent the
  // Cholesky decomposition from working
  Vec rmvn_robust(const Vec &Mu, const Spd &Sigma);
  Vec rmvn_L(const Vec &mu, const Mat &L);
  Vec rmvn_ivar(const Vec &Mu, const Spd &Sigma_Inverse);
  Vec rmvn_ivar_L(const Vec &Mu, const Mat &Ivar_chol);
  Vec rmvn_ivar_U(const Vec &Mu, const Mat &Ivar_chol_transpose);
  Vec rmvn_suf(const Spd & Ivar, const Vec & IvarMu);

  Vec rmvn_robust_mt(RNG & rng, const Vec &Mu, const Spd &Sigma);
  Vec rmvn_L_mt(RNG & rng, const Vec &mu, const Mat &L);
  Vec rmvn_ivar_mt(RNG & rng, const Vec &Mu, const Spd &Sigma_Inverse);
  Vec rmvn_ivar_L_mt(RNG & rng, const Vec &Mu, const Mat &Ivar_chol);
  Vec rmvn_ivar_U_mt(RNG & rng, const Vec &Mu, const Mat &Ivar_chol_transpose);
  Vec rmvn_suf_mt(RNG & rng, const Spd & Ivar, const Vec & IvarMu);


  double dmvn(const Vec &y, const Vec &mu, const Spd &Siginv,
	      double ldsi, bool logscale);
  double dmvn_zero_mean(const Vec &y, const Spd &Siginv,
			double ldsi, bool logscale);
  double dmvn(const Vec &y, const Vec &mu, const Spd &Siginv, bool logscale);

  // Y~ matrix_normal(Mu, Siginv, Ominv) if
  // Vec(Y) ~ N(Vec(Mu), (Siginv \otimes Ominv)^{-1})

  Mat rmatrix_normal_ivar(const Mat & Mu, const Spd &Siginv,
			     const Spd &Ominv);
  Mat rmatrix_normal_ivar_mt(RNG & rng, const Mat & Mu, const Spd &Siginv,
			     const Spd &Ominv);
  double dmatrix_normal_ivar(const Mat &Y, const Mat &Mu,
			     const Spd &Siginv, const Spd &Ominv,
			     bool logscale);
  double dmatrix_normal_ivar(const Mat &Y, const Mat &Mu,
			     const Spd &Siginv, double ldsi,
			     const Spd &Ominv, double ldoi,
			     bool logscale);

  //  uniforms shrinkage prior in usp.cpp
  double dusp(double x, double z0, bool logscale);
  double pusp(double x, double z0, bool logscale);
  double qusp(double p, double z0);

  double rusp(double z0);
  double rusp_mt(RNG & rng, double z0);

  //  Spd rWish( double,  Spd &);
  Spd rWish(double df, const Spd &sumsq_inv, bool inv=false);
  Spd rWish_mt(RNG &, double df, const Spd &sumsq_inv, bool inv=false);
  Spd rWishChol(double df, const Mat &sumsq_upper_chol, bool inv=false);
  Spd rWishChol_mt(RNG &, double df, const Mat &sumsq_upper_chol, bool inv=false);
  double dWish(const Spd &S, const Spd &sumsq, double df, bool logscale, bool inv=false);
  inline double dWishinv(const Spd &S, const Spd &sumsq, double df, bool logscale){
    return dWish(S, sumsq, df, logscale, true); }

  double ddirichlet(const Vec & x, const Vec & nu, bool logscale);
  double ddirichlet(const VectorView & x, const Vec & nu, bool logscale);
  double ddirichlet(const Vec & x, const VectorView & nu, bool logscale);
  double ddirichlet(const VectorView & x, const VectorView & nu, bool logscale);

  double ddirichlet(const Vec & x, const ConstVectorView & nu, bool logscale);
  double ddirichlet(const ConstVectorView & x, const Vec & nu, bool logscale);
  double ddirichlet(const ConstVectorView & x, const ConstVectorView & nu, bool logscale);

  double ddirichlet(const VectorView & x, const ConstVectorView & nu, bool logscale);
  double ddirichlet(const ConstVectorView & x, const VectorView & nu, bool logscale);

  Vec mdirichlet(const Vec &nu);
  double dirichlet_loglike(const Vec &nu, Vec *g, Mat *h,
			   const Vec & sumlogpi, double nobs);

  Vec rdirichlet(const Vec & nu);
  Vec rdirichlet_mt(RNG &rng, const Vec & nu);
  Vec rdirichlet(const VectorView & nu);
  Vec rdirichlet_mt(RNG &rng, const VectorView & nu);
  Vec rdirichlet(const ConstVectorView & nu);
  Vec rdirichlet_mt(RNG &rng, const ConstVectorView & nu);

  unsigned int rmulti(const Vec &);
  unsigned int rmulti(const VectorView &);
  unsigned int rmulti(const ConstVectorView &);
  unsigned int rmulti_mt(RNG &rng, const Vec &);
  unsigned int rmulti_mt(RNG &rng, const VectorView &);
  unsigned int rmulti_mt(RNG &rng, const ConstVectorView &);

  int rmulti(int, int);
  int rmulti_mt(RNG &, int, int);

  double dmvt(const Vec &x,  const Vec &mu, const Spd &Siginv,
	      double nu, double ldsi, bool logscale);
  double dmvt(const Vec &x,  const Vec &mu, const Spd &Siginv,
	      double nu, bool logscale);

  Vec rmvt(const Vec &mu, const Spd &Sigma, double nu);
  Vec rmvt_ivar(const Vec &mu, const Spd &Sigma, double nu);
  Vec rmvt_mt(RNG &, const Vec &mu, const Spd &Sigma, double nu);
  Vec rmvt_ivar_mt(RNG &, const Vec &mu, const Spd &Sigma, double nu);
}



#endif// BOOM_DISTRIBUTIONS_HPP
