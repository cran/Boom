/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#ifndef BOOM_ZERO_INFLATED_LOGNORMAL_MODEL_HPP_
#define BOOM_ZERO_INFLATED_LOGNORMAL_MODEL_HPP_

#include <Models/Policies/CompositeParamPolicy.hpp>
#include <Models/Policies/ConjugatePriorPolicy.hpp>
#include <Models/DoubleModel.hpp>
#include <Models/EmMixtureComponent.hpp>
#include <Models/BinomialModel.hpp>
#include <Models/GaussianModel.hpp>

namespace BOOM{

  class ZeroInflatedLognormalModel
      : public CompositeParamPolicy,
        public PriorPolicy,
        public DoubleModel,
        public EmMixtureComponent
  {
   public:
    ZeroInflatedLognormalModel();
    ZeroInflatedLognormalModel(const ZeroInflatedLognormalModel &rhs);
    virtual ZeroInflatedLognormalModel * clone()const;

    virtual double pdf(Ptr<Data>, bool logscale)const;
    virtual double pdf(const Data *, bool logscale)const;
    virtual double logp(double x)const;
    virtual double sim()const;

    // This model does not keep copies of the original data set, it
    // uses the sufficient statistics for its component
    // models. instead.
    virtual void add_data(Ptr<Data>);
    void add_data_raw(double y);
    virtual void add_mixture_data(Ptr<Data>, double weight);
    void add_mixture_data_raw(double y, double weight);
    virtual void clear_data();
    virtual void combine_data(const Model &, bool just_suf = true);

    virtual void find_posterior_mode();
    virtual void mle();
    void set_conjugate_prior(double normal_mean_guess,
                             double normal_mean_weight,
                             double normal_standard_deviation_guess,
                             double normal_standard_deviation_weight,
                             double nonzero_proportion_guess,
                             double nonzero_proportion_weight);

    // Mean and standard deviation of log of the positive observations.
    double mu()const;
    void set_mu(double mu);

    double sigma()const;
    void set_sigma(double sigma);
    void set_sigsq(double sigsq);

    // The probability that an event is greater than zero.
    double positive_probability()const;
    void set_positive_probability(double prob);

    // Moments of the actual observations, including zeros.
    double mean()const;
    double variance()const;
    double sd()const;

    Ptr<GaussianModel> Gaussian_model();
    Ptr<BinomialModel> Binomial_model();
   private:
    Ptr<GaussianModel> gaussian_;
    Ptr<BinomialModel> binomial_;
    double precision_;

    mutable double log_probability_of_positive_;
    mutable double log_probability_of_zero_;
    mutable bool log_probabilities_are_current_;
    boost::function<void(void)> create_binomial_observer();
    void observe_binomial_probability();
    void check_log_probabilities()const;
    Ptr<DoubleData> DAT(Ptr<Data>)const;
  };

}

#endif // BOOM_ZERO_INFLATED_LOGNORMAL_MODEL_HPP_
