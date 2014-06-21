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
#ifndef BOOM_WEIGHTED_MVN_MODEL_HPP
#define BOOM_WEIGHTED_MVN_MODEL_HPP

#include <LinAlg/Types.hpp>
#include <Models/ModelTypes.hpp>
#include <Models/SpdParams.hpp>
#include <Models/Sufstat.hpp>
#include <Models/Policies/ParamPolicy_2.hpp>
#include <Models/Policies/SufstatDataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/WeightedData.hpp>

namespace BOOM{

  class WeightedMvnSuf
    : public SufstatDetails<WeightedVectorData>
  {
  public:
    WeightedMvnSuf(uint p);
    WeightedMvnSuf(const WeightedMvnSuf &rhs);
    WeightedMvnSuf * clone()const;

    void clear();
    void Update(const WeightedVectorData &x);

    const Vec & sum()const;
    const Spd & sumsq()const;
    double n()const;
    double sumw()const;
    double sumlogw()const;

    Vec ybar()const;
    Spd var_hat()const;
    Spd center_sumsq(const Vec &mu)const;
    Spd center_sumsq()const;
    void combine(Ptr<WeightedMvnSuf>);
    void combine(const WeightedMvnSuf &);
    WeightedMvnSuf * abstract_combine(Sufstat *s);

    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true);
    virtual ostream &print(ostream &out)const;
  private:
    Vec sum_;
    Spd sumsq_;
    double n_;
    double sumw_;
    double sumlogw_;
  };

  class WeightedMvnModel
    : public ParamPolicy_2<VectorParams, SpdParams>,
      public SufstatDataPolicy<WeightedVectorData, WeightedMvnSuf>,
      public PriorPolicy,
      public LoglikeModel
  {
  public:
    WeightedMvnModel(uint p, double mu=0.0, double sig=1.0);
    WeightedMvnModel(const Vec &mean, const Spd &Var); // N(mu, Var)
    WeightedMvnModel(const WeightedMvnModel &m);
    WeightedMvnModel *clone() const;

    Ptr<VectorParams> Mu_prm();
    const Ptr<VectorParams> Mu_prm()const;
    Ptr<SpdParams> Sigma_prm();
    const Ptr<SpdParams> Sigma_prm()const;

    const Vec & mu() const;
    const Spd & Sigma()const;
    const Spd & siginv() const;
    double ldsi()const;

    void set_mu(const Vec &);
    void set_Sigma(const Spd &);
    void set_siginv(const Spd &);
    void mle();
    double loglike() const;

    double pdf(Ptr<Data>, bool logscale)const;
    double pdf(Ptr<DataType>, bool logscale)const;
  };
}
#endif// BOOM_WEIGHTED_MVN_MODEL_HPP
