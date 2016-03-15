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
#include <Models/MvnGivenSigma.hpp>
#include <distributions.hpp>
#include <utility>
#include <Models/SpdData.hpp>
namespace BOOM{

  typedef MvnGivenSigma MGS;

  MGS::MvnGivenSigma(const Vector &b, double k, const SpdMatrix & siginv)
    : ParamPolicy(new VectorParams(b), new UnivParams(k)),
      DataPolicy(new MvnSuf(b.size())),
      PriorPolicy(),
      Sigma_(new SpdData(siginv, true))
  {
  }

  MGS::MvnGivenSigma(const Vector &b, double k)
    : ParamPolicy(new VectorParams(b), new UnivParams(k)),
      DataPolicy(new MvnSuf(b.size())),
      PriorPolicy()
  {
    // Sigma must be set before the class can be used_;
  }

  MGS::MvnGivenSigma(const Vector &b, double k, Ptr<SpdData> Sigma)
    : ParamPolicy(new VectorParams(b), new UnivParams(k)),
      DataPolicy(new MvnSuf(b.size())),
      PriorPolicy(),
      Sigma_(Sigma)
  {
  }

  MGS::MvnGivenSigma(Ptr<VectorParams> mu, Ptr<UnivParams> Kappa)
    : ParamPolicy(mu, Kappa),
      DataPolicy(new MvnSuf(mu->size())),
      PriorPolicy()
  {
    // Sigma must be set before the class can be used_;
  }

  MGS::MvnGivenSigma(Ptr<VectorParams> mu, Ptr<UnivParams> Kappa,
		     Ptr<SpdData> Sigma)
    : ParamPolicy(mu, Kappa),
      DataPolicy(new MvnSuf(mu->size())),
      PriorPolicy(),
      Sigma_(Sigma)
  {}

  MGS::MvnGivenSigma(const MGS & rhs)
    : Model(rhs),
      VectorModel(rhs),
      MvnBase(rhs),
      LoglikeModel(rhs),
      ParamPolicy(rhs),
      DataPolicy(rhs),
      PriorPolicy(rhs),
      Sigma_(rhs.Sigma_)
  {
  }

  MGS * MGS::clone()const{return new MGS(*this);}

  void MGS::set_Sigma(Ptr<SpdData> S){ Sigma_ = S;}
  void MGS::set_Sigma(const SpdMatrix &V, bool ivar){
    NEW(SpdData, d)(V, ivar);
    this->set_Sigma(d);
  }

  Ptr<VectorParams> MGS::Mu_prm(){
    return ParamPolicy::prm1();}
  const Ptr<VectorParams> MGS::Mu_prm()const{
    return ParamPolicy::prm1();}

  Ptr<UnivParams> MGS::Kappa_prm(){
    return ParamPolicy::prm2();}
  const Ptr<UnivParams> MGS::Kappa_prm()const{
    return ParamPolicy::prm2();}

  uint MGS::dim()const{return mu().size();}
  const Vector & MGS::mu()const{ return Mu_prm()->value();}
  double MGS::kappa()const{return Kappa_prm()->value();}

  void MGS::set_mu(const Vector &v){Mu_prm()->set(v);}
  void MGS::set_kappa(double kap){Kappa_prm()->set(kap);}

  void MGS::mle(){
    check_Sigma();
    set_mu(suf()->ybar());
    double np = suf()->n() * dim();
    double ss = traceAB(Sigma_->ivar(), suf()->center_sumsq());
    set_kappa(np/ss);
  }

  double MGS::loglike(const Vector &mu_kappa)const{
    check_Sigma();
    const ConstVectorView mu(mu_kappa, 0, dim());
    double kappa = mu_kappa.back();
    return MvnBase::log_likelihood(
        mu,
        Sigma_->ivar() * kappa,
        *suf());
  }

  double MGS::pdf(Ptr<Data> dp, bool logsc)const{
    return this->pdf(DAT(dp), logsc);}
  double MGS::pdf(Ptr<DataType> dp, bool logsc)const{
    check_Sigma();
    double k = kappa();
    double ldsi = Sigma_->ldsi();
    return dmvn(dp->value(), mu(), Sigma_->ivar() * k, ldsi + dim()*log(k),
                logsc);
  }

  void MGS::check_Sigma()const{
    if(!!Sigma_) return;
    ostringstream err;
    err << "Sigma has not been set in instance of MvnGivenSigma."
	<< endl;
    report_error(err.str());
  }

  const SpdMatrix & MGS::Sigma()const{
    S = Sigma_->value()/kappa();
    return S;
  }

  const SpdMatrix & MGS::siginv()const{
    S = Sigma_->ivar() * kappa();
    return S;
  }

  double MGS::ldsi()const{
    double ans = Sigma_->ldsi();
    ans += dim()*log(kappa());
    return ans;
  }

  double MGS::Logp(const Vector &x, Vector &g, Matrix &h, uint nd)const{
    const SpdMatrix & siginv(this->siginv());
    const Vector & mu(this->mu());
    double ans = dmvn(x,mu, siginv, ldsi(), true);
    if(nd>0){
      g = -(siginv * (x-mu));
      if(nd>1) h = -siginv;}
    return ans;
  }

  Vector MGS::sim()const{ return rmvn_ivar(mu(),siginv());  }

}
