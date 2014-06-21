/*
  Copyright (C) 2007 Steven L. Scott

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
#ifndef BOOM_BINOMIAL_MODEL_HPP
#define BOOM_BINOMIAL_MODEL_HPP

#include <Models/ModelTypes.hpp>
#include <Models/Sufstat.hpp>
#include <Models/EmMixtureComponent.hpp>
#include <Models/Policies/ParamPolicy_1.hpp>
#include <Models/Policies/SufstatDataPolicy.hpp>
#include <Models/Policies/ConjugatePriorPolicy.hpp>

namespace BOOM{

  class BetaBinomialSampler;

  class BinomialSuf : public SufstatDetails<IntData>{
  public:
    BinomialSuf();
    BinomialSuf(const BinomialSuf &rhs);
    BinomialSuf * clone()const;
    void set(double sum, double observation_count);

    double sum()const;
    double nobs()const;
    virtual void clear();
    virtual void Update(const IntData &);
    void update_raw(double y);
    void batch_update(double n, double y);

    void add_mixture_data(double y, double prob);

    BinomialSuf * abstract_combine(Sufstat *s);
    void combine(Ptr<BinomialSuf>);
    void combine(const BinomialSuf &);

    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true);
    virtual ostream &print(ostream &out)const;
  private:
    double sum_, nobs_;
  };

  class BinomialModel
    : public ParamPolicy_1<UnivParams>,
      public SufstatDataPolicy<IntData,BinomialSuf>,
      public ConjugatePriorPolicy<BetaBinomialSampler>,
      public NumOptModel,
      public EmMixtureComponent
  {
  public:
    BinomialModel(uint n=1, double p=.5);
    BinomialModel(const BinomialModel &rhs);
    BinomialModel * clone()const;

    virtual void mle();
    virtual double Loglike(Vec &g, Mat &h, uint nd)const;

    uint n()const;
    double prob()const;
    void set_prob(double p);

    virtual double pdf(Ptr<Data> x, bool logscale)const;
    virtual double pdf(const Data * x, bool logscale)const;
    double pdf(uint x, bool logscale)const;

    Ptr<UnivParams> Prob_prm();
    const Ptr<UnivParams> Prob_prm()const;
    uint sim()const;

    virtual void find_posterior_mode();
    virtual void add_mixture_data(Ptr<Data>, double prob);

    // Sets the prior distribution to a Beta(a, b), and sets the prior
    // sampler to a BetaBinomialSampler.
    void set_conjugate_prior(double a, double b);
    void set_conjugate_prior(Ptr<BetaBinomialSampler>);
  private:
    const uint n_;
  };

}

#endif // BOOM_BINOMIAL_MODEL_HPP
