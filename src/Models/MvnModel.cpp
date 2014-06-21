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
#include <Models/MvnModel.hpp>
#include <LinAlg/Vector.hpp>
#include <LinAlg/SpdMatrix.hpp>
#include <LinAlg/Types.hpp>
#include <cmath>
#include <distributions.hpp>

#include <Models/MvnGivenSigma.hpp>
#include <Models/WishartModel.hpp>

#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/PosteriorSamplers/MvnConjSampler.hpp>

#include <boost/bind.hpp>

namespace BOOM{

  double MvnModel::loglike()const{
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

  void MvnModel::add_raw_data(const Vec &y){
    NEW(VectorData, dp)(y);
    this->add_data(dp);
  }

  double MvnModel::pdf(Ptr<Data> dp, bool logscale)const{
    double ans = logp(DAT(dp)->value());
    return logscale ? ans : exp(ans);
  }

  double MvnModel::pdf(const Data * dp, bool logscale)const{
    double ans = logp(DAT(dp)->value());
    return logscale ? ans : exp(ans);
  }

  double MvnModel::pdf(const Vec &x, bool logscale)const{
    double ans = logp(x);
    return logscale ? ans : exp(ans);
  }

  Vec MvnModel::sim()const{
    return rmvn(mu(), Sigma());
  }

  void MvnModel::set_conjugate_prior(Ptr<MvnGivenSigma> Mu,
				     Ptr<WishartModel> Siginv){
    Mu->set_Sigma(Sigma_prm());
    NEW(MvnConjSampler, pri)(this, Mu,Siginv);
    set_conjugate_prior(pri);
  }

  void MvnModel::set_conjugate_prior(Ptr<MvnConjSampler> pri){
    ConjPriorPolicy::set_conjugate_prior(pri); }


  //======================================================================

  MvnModel::MvnModel(uint p, double mu, double sigma)
    : Base(p,mu,sigma),
      DataPolicy(new MvnSuf(p))
  {}

  MvnModel::MvnModel(const Vec &mean, const Spd &Var, bool ivar) // N(mu, Var)
    : Base(mean,Var, ivar),
      DataPolicy(new MvnSuf(mean.size()))
  {}

  MvnModel::MvnModel(Ptr<VectorParams> mu, Ptr<SpdParams> Sigma)
    : Base(mu, Sigma),
      DataPolicy(new MvnSuf(mu->dim()))
  {}

  MvnModel::MvnModel(const std::vector<Vec> &v)       // N(mu.hat, V.hat)
    : Base(v[0].size()),
      DataPolicy(new MvnSuf(v[0].size())),
      ConjPriorPolicy()
  {
    set_data_raw(v.begin(), v.end());
    mle();
  }


  MvnModel::MvnModel(const MvnModel &rhs)
    : Model(rhs),
      VectorModel(rhs),
      MLE_Model(rhs),
      Base(rhs),
      LoglikeModel(rhs),
      DataPolicy(rhs),
      ConjPriorPolicy(rhs),
      EmMixtureComponent(rhs)
  {}

  MvnModel * MvnModel::clone() const{return new MvnModel(*this);}

  //  const Ptr<MvnSuf> MvnModel::suf()const{ return DataPolicy::suf();}

//   Ptr<VectorParams> MvnModel::Mu_prm(){
//     return ParamPolicy::prm1();}
//   const Ptr<VectorParams> MvnModel::Mu_prm()const{
//     return ParamPolicy::prm1();}

//   Ptr<SpdParams> MvnModel::Sigma_prm(){
//     return ParamPolicy::prm2();}
//   const Ptr<SpdParams> MvnModel::Sigma_prm()const{
//     return ParamPolicy::prm2();}

//   const Vec & MvnModel::mu()const{return Mu_prm()->value();}
//   const Spd & MvnModel::Sigma()const{return Sigma_prm()->var();}
//   const Spd & MvnModel::siginv()const{return Sigma_prm()->ivar();}
//   double MvnModel::ldsi()const{return Sigma_prm()->ldsi();}

//   void MvnModel::set_mu(const Vec &v){Mu_prm()->set(v);}
//   void MvnModel::set_Sigma(const Spd &s){Sigma_prm()->set_var(s);}
//   void MvnModel::set_siginv(const Spd &ivar){Sigma_prm()->set_ivar(ivar);}
//   void MvnModel::set_S_Rchol(const Vec &sd, const Mat &L){
//     Sigma_prm()->set_S_Rchol(sd,L); }

  void MvnModel::mle(){
    set_mu(suf()->ybar());
    set_Sigma(suf()->var_hat());
  }

  void MvnModel::add_mixture_data(Ptr<Data> dp, double prob){
    suf()->add_mixture_data( DAT(dp)->value(), prob);
  }

  //======================================================================
}
