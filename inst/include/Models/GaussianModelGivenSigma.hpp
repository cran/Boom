/*
  Copyright (C) 2006 Steven L. Scott

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
#ifndef BOOM_GAUSSIAN_MODEL_GIVEN_SIGMA_HPP
#define BOOM_GAUSSIAN_MODEL_GIVEN_SIGMA_HPP

#include <Models/ModelTypes.hpp>
#include <Models/ParamTypes.hpp>
#include <Models/Sufstat.hpp>
#include <Models/Policies/SufstatDataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/Policies/ParamPolicy_2.hpp>

#include <Models/GaussianModelBase.hpp>

namespace BOOM{

  //------------------------------------------------------------

  class GaussianModelGivenSigma
    : public ParamPolicy_2<UnivParams, UnivParams>,
      public PriorPolicy,
      public SufstatDataPolicy<DoubleData, GaussianSuf>,
      public DiffDoubleModel,
      public NumOptModel
  {
    // mu ~ N(mu0, sigma^2/kappa)
    // parameters are mu0 and kappa
    // conjugate prior is normal for mu0 and gamma for kappa

  public:
    GaussianModelGivenSigma(Ptr<UnivParams> sigsq, double mu0=0,
			    double kappa=1);
    GaussianModelGivenSigma * clone()const override;

    void set_sigsq(Ptr<UnivParams>);
    void set_params(double mu0, double kappa);
    void set_mu(double mu0);
    void set_kappa(double kappa);

    double ybar()const;
    double sample_var()const;

    double mu()const;
    double kappa()const;
    double sigsq()const;
    double var()const;

    Ptr<UnivParams> Mu_prm();
    Ptr<UnivParams> Kappa_prm();
    const Ptr<UnivParams> Mu_prm()const;
    const Ptr<UnivParams> Kappa_prm()const;

    void mle() override;

    double Logp(double x, double &g, double &h, uint nd)const override;
    double Logp(const Vector & x, Vector &g, Matrix &h, uint nd)const;
    double Loglike(const Vector &mu_kappa, Vector &g, Matrix &h, uint nd)const override;

    void set_semi_conj_prior(double mu0, double v_mu0,
			     double pdf, double sigma_guess);
    void set_conj_prior(double mu0, double mu_n,
			double pdf, double pss);
    double sim()const override;

    void add_data_raw(double x);

  private:
    Ptr<UnivParams> sigsq_;
  };

}
#endif// BOOM_GAUSSIAN_MODEL_GIVEN_SIGMA_HPP
