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
#ifndef BOOM_MULTINOMIAL_MODEL_HPP
#define BOOM_MULTINOMIAL_MODEL_HPP
#include <Models/ModelTypes.hpp>
#include <Models/ParamTypes.hpp>
#include <Models/Sufstat.hpp>
#include <Models/CategoricalData.hpp>
#include <Models/Policies/SufstatDataPolicy.hpp>
#include <Models/Policies/ConjugatePriorPolicy.hpp>
#include <Models/Policies/ParamPolicy_1.hpp>
#include <Models/EmMixtureComponent.hpp>

namespace BOOM{

  class MultinomialSuf
    : public SufstatDetails<CategoricalData>
  {
  public:
    MultinomialSuf(uint p);
    MultinomialSuf(const MultinomialSuf &rhs);
    MultinomialSuf* clone()const;

    void Update(const CategoricalData &d);
    void add_mixture_data(uint y, double prob);
    void add_mixture_data(const Vector &weights);
    void update_raw(uint k);
    void clear();

    const Vec &n()const;
    void combine(Ptr<MultinomialSuf>);
    void combine(const MultinomialSuf &);
    MultinomialSuf * abstract_combine(Sufstat *s);

    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true);
    virtual ostream &print(ostream &out)const;
  private:
    Vec counts;
  };

  //======================================================================
  class MultinomialDirichletSampler;
  class DirichletModel;

  class MultinomialModel
    : public ParamPolicy_1<VectorParams>,
      public SufstatDataPolicy<CategoricalData, MultinomialSuf>,
      public ConjugatePriorPolicy<MultinomialDirichletSampler>,
      public LoglikeModel,
      public EmMixtureComponent
  {
  public:
    MultinomialModel(uint Nlevels);
    MultinomialModel(const Vec &probs );

    // The argument is a vector of names to use for factor levels to
    // be modeled.
    MultinomialModel(const std::vector<string> &);

    template <class Fwd> // iterator promotable to uint
    MultinomialModel(Fwd b, Fwd e);
    MultinomialModel(const MultinomialModel &rhs);
    MultinomialModel * clone()const;

    uint nlevels()const;
    Ptr<VectorParams> Pi_prm();
    const Ptr<VectorParams> Pi_prm()const;

    const double & pi(int s) const;
    const Vec & pi()const;
    void set_pi(const Vec &probs);

    uint size()const;         // number of potential outcomes;
    double loglike()const;
    void mle();
    double pdf(const Data * dp, bool logscale) const;
    double pdf(Ptr<Data> dp, bool logscale) const;
    void add_mixture_data(Ptr<Data>, double prob);

    void set_conjugate_prior(const Vec &nu);
    void set_conjugate_prior(Ptr<DirichletModel>);
    void set_conjugate_prior(Ptr<MultinomialDirichletSampler>);

    uint simdat()const;
   private:
    mutable Vec logp_;
    mutable bool logp_current_;
    void observe_logp();
    void set_observer();
    void check_logp()const;
  };

  template <class Fwd> // iterator promotable to uint
  MultinomialModel::MultinomialModel(Fwd b, Fwd e)
    : ParamPolicy(new VectorParams(1)),
      DataPolicy(new MultinomialSuf(1)),
      ConjPriorPolicy()
  {
    std::vector<uint> uivec(b,e);
    std::vector<Ptr<CategoricalData> >
      dvec(make_catdat_ptrs(uivec));

    uint nlev= dvec[0]->nlevels();
    Vec probs(nlev, 1.0/nlev);
    set_pi(probs);

    set_data(dvec);
    mle();
  }
}
#endif // BOOM_MULTINOMIAL_MODEL_HPP
