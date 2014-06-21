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
#ifndef BOOM_GLM_MVN_PRIOR_HPP
#define BOOM_GLM_MVN_PRIOR_HPP

#include <LinAlg/Types.hpp>
#include <Models/ModelTypes.hpp>
#include <Models/Sufstat.hpp>
#include <Models/Policies/ParamPolicy_2.hpp>
#include <Models/Policies/SufstatDataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/Glm/GlmCoefs.hpp>
#include <Models/MvnBase.hpp>

/***************************************************************************
 * GlmMvnPrior is the a prior for the conditional distribution of
 * GlmCoefs given a set of 0/1 indicator variables gamma.
 ***************************************************************************/

namespace BOOM{

  class GlmMvnSuf
    : public SufstatDetails<GlmCoefs>
  {
  public:
    GlmMvnSuf(uint p=0);
    GlmMvnSuf(const std::vector<Ptr<GlmCoefs> > & d);
    GlmMvnSuf * clone()const;

    virtual void clear();
    void Update(const GlmCoefs &beta);

    Spd center_sumsq(const Vec &b)const;
    const Vec & vnobs()const;  // sum of gamma
    const Spd & GTG()const;    // sum of gamma gamma^T
    const Mat & BTG()const;    // sum of beta gamma^T
    uint nobs()const;
    void combine(Ptr<GlmMvnSuf>);
    void combine(const GlmMvnSuf &);
    GlmMvnSuf * abstract_combine(Sufstat *s);

    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true);
    virtual ostream &print(ostream &out)const;
  private:
    mutable Spd bbt_;    // sum of beta beta.transpose()
    mutable Spd ggt_;    // sum of gamma * gamma^T
    Mat bgt_;    // sum of beta gamma.transpose()
    Vec vnobs_;  // sum of gamma
    uint nobs_;  // number of observations

    Vec b, gam;
    mutable bool sym_;
    void make_symmetric()const;
  };

  //______________________________________________________________________

  class GlmMvnPrior
    : public MvnBaseWithParams,
      public LoglikeModel,
      public SufstatDataPolicy<GlmCoefs,GlmMvnSuf>,
      public PriorPolicy
  {
    // conditional model for GlmCoefs given gamma.
    // beta | gamma ~ N( mu()_gamma, siginv_gamma)

    typedef MvnBaseWithParams Base;
  public:
    GlmMvnPrior(uint p, double mu=0, double sig=1.0);
    GlmMvnPrior(const Vec &mean, const Spd &V, bool ivar=false);
    GlmMvnPrior(Ptr<VectorParams> mu, Ptr<SpdParams> Sigma);
    GlmMvnPrior(const std::vector<Ptr<GlmCoefs> > &);
    GlmMvnPrior(const GlmMvnPrior &rhs);
    GlmMvnPrior * clone()const;

    void mle();
    double loglike()const;
    double pdf(Ptr<Data>, bool logscale)const;
    double pdf(const Ptr<GlmCoefs> x, bool logscale)const;
    const Mat & siginv_chol()const;

    Vec simulate(const Selector &)const;

    Spd sumsq()const;
  private:
    double eval_logdet(const Selector &inc, const Mat &LT)const;
    virtual Vec sim()const;
    mutable Spd ivar_;
    mutable Vec wsp_;
  };
}
#endif// BOOM_GLM_MVN_PRIOR_HPP
