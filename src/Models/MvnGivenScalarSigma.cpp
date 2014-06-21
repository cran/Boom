/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#include <distributions.hpp>
#include <Models/MvnGivenScalarSigma.hpp>

namespace BOOM{

  typedef MvnGivenScalarSigma MGSS;

  MvnGivenScalarSigma::MvnGivenScalarSigma(const Spd &ominv,
                                           Ptr<UnivParams> sigsq)
      : ParamPolicy(new VectorParams(nrow(ominv))),
      DataPolicy(new MvnSuf(nrow(ominv))),
      PriorPolicy(),
      sigsq_(sigsq),
      omega_(ominv, true),
      wsp_(ominv)
      {}

  MvnGivenScalarSigma::MvnGivenScalarSigma(const Vec &mean,
                                           const Spd &ominv,
                                           Ptr<UnivParams> sigsq)
      : ParamPolicy(new VectorParams(mean)),
      DataPolicy(new MvnSuf(mean.size())),
      PriorPolicy(),
      sigsq_(sigsq),
      omega_(ominv, true),
      wsp_(mean.size())
      {}


  MGSS::MvnGivenScalarSigma(const MGSS &rhs)
      : Model(rhs),
        VectorModel(rhs),
        MLE_Model(rhs),
        MvnBase(rhs),
        LoglikeModel(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        sigsq_(rhs.sigsq_),
        omega_(rhs.omega_),
        wsp_(rhs.wsp_)
      {}

  MGSS * MGSS::clone()const{return new MGSS(*this);}

  Ptr<VectorParams> MGSS::Mu_prm(){return ParamPolicy::prm();}
  const Ptr<VectorParams> MGSS::Mu_prm()const{return ParamPolicy::prm();}

  uint MGSS::dim()const{ return nrow(wsp_); }
  const Vec & MGSS::mu()const{ return Mu_prm()->value(); }

  const Spd & MGSS::Sigma()const{
    wsp_ = omega_.var() * sigsq_->value();
    return wsp_;
  }

  const Spd & MGSS::siginv()const{
    wsp_ = omega_.ivar() / sigsq_->value();
    return wsp_;
  }

  double MGSS::ldsi()const{
    return omega_.ldsi() - dim() * log(sigsq_->value());
  }

  const Spd & MGSS::Omega()const{ return omega_.var();}
  const Spd & MGSS::ominv()const{ return omega_.ivar();}
  double MGSS::ldoi()const{ return omega_.ldsi();}

  void MGSS::set_mu(const Vec &mu){Mu_prm()->set(mu);}
  void MGSS::mle(){ set_mu(suf()->ybar()); }

  // copied from MvnModel::loglike.  Consider moving to MvnBase
  double MGSS::loglike()const{
    const double log2pi = 1.83787706641;
    double dim = mu().size();
    double n = suf()->n();
    const Vec ybar = suf()->ybar();
    const Spd sumsq = suf()->center_sumsq();

    double qform = n*(siginv().Mdist(ybar, mu()));
    qform+= traceAB(siginv(), sumsq);

    double nc = 0.5*n*( -dim*log2pi + ldsi());

    double ans = nc - .5*qform;
    return ans;
  }

  double MGSS::pdf(Ptr<Data> dp, bool logscale)const{
    const Vec &y(DAT(dp)->value());
    return dmvn(y, mu(), siginv(), ldsi(), logscale);
  }

}// namespace BOOM
