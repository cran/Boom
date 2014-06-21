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
#ifndef BOOM_GAUSSIAN_MODEL_H
#define BOOM_GAUSSIAN_MODEL_H

#include <Models/GaussianModelBase.hpp>
#include <Models/ParamTypes.hpp>
#include <Models/Policies/ConjugatePriorPolicy.hpp>
#include <Models/Policies/ParamPolicy_2.hpp>
#include <Models/EmMixtureComponent.hpp>
#include <Models/PosteriorSamplers/GaussianConjSampler.hpp>

namespace BOOM{
  class GaussianModelGivenSigma;
  class GammaModel;
  class GaussianConjSampler;

  //------------------------------------------------------------
  class GaussianModel
      : public GaussianModelBase,
        public ParamPolicy_2<UnivParams, UnivParams>,
        public ConjugatePriorPolicy<GaussianConjSampler>
  {
  public:
    GaussianModel();  // N(0,1)
    GaussianModel(double mean, double sd);
    GaussianModel(const std::vector<double> &v);
    GaussianModel(const GaussianModel &rhs);
    GaussianModel * clone()const;

    void set_params(double Mean, double Var);
    void set_mu(double m);
    void set_sigsq(double s);

    virtual double mu()const;
    virtual double sigsq()const;
    virtual double sigma()const;

    Ptr<UnivParams> Mu_prm();
    Ptr<UnivParams> Sigsq_prm();
    const Ptr<UnivParams> Mu_prm()const;
    const Ptr<UnivParams> Sigsq_prm()const;

    virtual void mle();

    void set_conjugate_prior(double mu0, double kappa,
			     double df, double sigma_guess);
    void set_conjugate_prior(Ptr<GaussianModelGivenSigma>,
			     Ptr<GammaModel>);
    void set_conjugate_prior(Ptr<GaussianConjSampler>);

    double Loglike(Vec &g, Mat &h, uint nd)const;

    virtual void find_posterior_mode();
  };

}
#endif// BOOM_GAUSSIAN_MODEL_H
