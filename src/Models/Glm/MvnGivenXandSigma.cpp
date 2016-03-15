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
#include <Models/Glm/MvnGivenXandSigma.hpp>
#include <distributions.hpp>

namespace BOOM{

  typedef MvnGivenXandSigma MGXS;

  MGXS::MvnGivenXandSigma(RegressionModel * mod, Ptr<VectorParams> Mu,
			  Ptr<UnivParams> prior_ss, double diag_wgt)
    : ParamPolicy(Mu, prior_ss),
      mod_(mod),
      ivar_(new SpdParams(Mu->dim())),
      Lambda_(Mu->dim()),
      diagonal_weight_(diag_wgt),
      current_(false)
  {
    assert(diagonal_weight_>=0 && diagonal_weight_<=1);
  }


  MGXS::MvnGivenXandSigma(RegressionModel * mod, Ptr<VectorParams> Mu,
			  Ptr<UnivParams> prior_ss,
			  const Vector & Lambda, double diag_wgt)
    : ParamPolicy(Mu, prior_ss),
      mod_(mod),
      ivar_(new SpdParams(Mu->dim())),
      Lambda_(Lambda),
      diagonal_weight_(diag_wgt),
      current_(false)
  {
    assert(Lambda_.size()==Mu->dim());
    assert(diagonal_weight_>=0 && diagonal_weight_<=1);
  }

  MGXS::MvnGivenXandSigma(const MGXS & rhs)
    : Model(rhs),
      VectorModel(rhs),
      MvnBase(rhs),
      ParamPolicy(rhs),
      DataPolicy(rhs),
      PriorPolicy(rhs),
      mod_(rhs.mod_),
      ivar_(rhs.ivar_->clone()),
      Lambda_(rhs.Lambda_),
      diagonal_weight_(rhs.diagonal_weight_),
      current_(rhs.current_)
  {}

  MGXS * MGXS::clone()const{return new MGXS(*this);}

  const Vector & MGXS::mu()const{return Mu_prm()->value();}

  double MGXS::prior_sample_size()const{
    return Kappa_prm()->value();
  }

  const SpdMatrix & MGXS::Sigma()const{
    set_ivar();
    return ivar_->var();
  }

  const SpdMatrix & MGXS::siginv()const{
    set_ivar();
    return ivar_->ivar();
  }

  double MGXS::ldsi()const{
    set_ivar();
    return ivar_->ldsi();
  }

  double MGXS::pdf(Ptr<Data> dp, bool logscale)const{
    Ptr<GlmCoefs> d(DAT(dp));
    double ans =  logp(d->Beta());
    return logscale ? ans : exp(ans);
  }

  void MGXS::set_ivar()const{
    if(current_) return;

    SpdMatrix ivar(mod_->xtx());
    double w = diagonal_weight_;

    if(w>= 1.0) ivar.set_diag(ivar.diag());
    else if(w>0.0){
      ivar *= (1-w);
      ivar.diag()/=(1-w);
    }

    ivar.diag()+= Lambda_;
    double sigsq = mod_->sigsq();
    ivar/=sigsq;

    ivar_->set_ivar(ivar);
    current_ = true;
  }

  Ptr<VectorParams> MGXS::Mu_prm(){
    return ParamPolicy::prm1();}
  const Ptr<VectorParams> MGXS::Mu_prm()const{
    return ParamPolicy::prm1();}

  Ptr<UnivParams> MGXS::Kappa_prm(){
    return ParamPolicy::prm2();}
  const Ptr<UnivParams> MGXS::Kappa_prm()const{
    return ParamPolicy::prm2();}

  Vector MGXS::sim()const{
    const Matrix & L(ivar_->var_chol());
    uint p = dim();
    Vector ans(p);
    for(uint i=0; i<p; ++i) ans[i] = rnorm();
    ans = L * ans;
    ans += mu();
    return ans;
  }
}
