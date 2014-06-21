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

#ifndef MVN_MODEL_H
#define MVN_MODEL_H

#include <LinAlg/Types.hpp>
#include <Models/ModelTypes.hpp>
#include <Models/Sufstat.hpp>
#include <Models/Policies/ParamPolicy_2.hpp>
#include <Models/Policies/SufstatDataPolicy.hpp>
#include <Models/Policies/ConjugatePriorPolicy.hpp>
#include <Models/EmMixtureComponent.hpp>
#include <Models/MvnBase.hpp>
#include <Models/MvnGivenSigma.hpp>
#include <Models/PosteriorSamplers/MvnConjSampler.hpp>
#include <Models/WishartModel.hpp>

namespace BOOM{

  class MvnModel:
    public MvnBaseWithParams,
    public LoglikeModel,
    public SufstatDataPolicy<VectorData, MvnSuf>,
    public ConjugatePriorPolicy<MvnConjSampler>,
    public EmMixtureComponent
  {
  public:
    typedef MvnBaseWithParams Base;
    MvnModel(uint p, double mu=0.0, double sig=1.0);   // N(mu.1, diag(sig^2))
    MvnModel(const Vec &mean, const Spd &V,      // N(mu,V)... if(ivar) then V
	     bool ivar=false);                   // is the inverse variance.
    MvnModel(Ptr<VectorParams> mu, Ptr<SpdParams> Sigma);
    MvnModel(const std::vector<Vec> &v);       // N(mu.hat, V.hat)
    MvnModel(const MvnModel &m);
    MvnModel *clone() const;

    virtual void mle();
    virtual void add_mixture_data(Ptr<Data>, double prob);
    double loglike() const;

    void add_raw_data(const Vec &y);
    double pdf(Ptr<Data>, bool logscale)const;
    double pdf(const Data *, bool logscale)const;
    double pdf(const Vec &x, bool logscale)const;

    void set_conjugate_prior(Ptr<MvnGivenSigma>, Ptr<WishartModel>);
    void set_conjugate_prior(Ptr<MvnConjSampler>);

    Vec sim()const;
  };
  //------------------------------------------------------------
}
#endif  // MVN_MODEL_H
