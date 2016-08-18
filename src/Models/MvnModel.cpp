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

#include <cmath>
#include <distributions.hpp>

#include <Models/MvnGivenSigma.hpp>
#include <Models/WishartModel.hpp>

#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/PosteriorSamplers/MvnConjSampler.hpp>

#include <boost/bind.hpp>

namespace BOOM{

  double MvnModel::loglike(const Vector &mu_siginv)const{
    const ConstVectorView mu(mu_siginv, 0, dim());
    SpdMatrix siginv(dim());
    Vector::const_iterator b = mu_siginv.cbegin() + dim();
    siginv.unvectorize(b, true);
    return MvnBase::log_likelihood(mu, siginv, *suf());
  }

  void MvnModel::add_raw_data(const Vector &y){
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

  double MvnModel::pdf(const Vector &x, bool logscale)const{
    double ans = logp(x);
    return logscale ? ans : exp(ans);
  }

  Vector MvnModel::sim()const{
    return sim(GlobalRng::rng);
  }

  Vector MvnModel::sim(RNG &rng)const{
    return rmvn_L_mt(rng, mu(), Sigma_chol());
  }

  void MvnModel::set_conjugate_prior(Ptr<MvnGivenSigma> Mu,
                                     Ptr<WishartModel> Siginv){
    Mu->set_Sigma(Sigma_prm());
    NEW(MvnConjSampler, pri)(this, Mu,Siginv);
    set_conjugate_prior(pri);
  }

  void MvnModel::set_conjugate_prior(Ptr<MvnConjSampler> pri){
    set_method(pri);
  }

  //======================================================================

  MvnModel::MvnModel(uint p, double mu, double sigma)
    : Base(p,mu,sigma),
      DataPolicy(new MvnSuf(p))
  {}

  MvnModel::MvnModel(const Vector &mean, const SpdMatrix &Var, bool ivar)
    : Base(mean,Var, ivar),
      DataPolicy(new MvnSuf(mean.size()))
  {}

  MvnModel::MvnModel(Ptr<VectorParams> mu, Ptr<SpdParams> Sigma)
    : Base(mu, Sigma),
      DataPolicy(new MvnSuf(mu->dim()))
  {}

  MvnModel::MvnModel(const std::vector<Vector> &v)
    : Base(v[0].size()),
      DataPolicy(new MvnSuf(v[0].size())),
      PriorPolicy()
  {
    set_data_raw(v.begin(), v.end());
    mle();
  }


  MvnModel::MvnModel(const MvnModel &rhs)
    : Model(rhs),
      VectorModel(rhs),
      Base(rhs),
      LoglikeModel(rhs),
      DataPolicy(rhs),
      PriorPolicy(rhs),
      EmMixtureComponent(rhs)
  {}

  MvnModel * MvnModel::clone() const{return new MvnModel(*this);}

  void MvnModel::mle(){
    set_mu(suf()->ybar());
    set_Sigma(suf()->var_hat());
  }

  void MvnModel::initialize_params() {
    mle();
  }

  void MvnModel::add_mixture_data(Ptr<Data> dp, double prob){
    suf()->add_mixture_data( DAT(dp)->value(), prob);
  }

  //======================================================================
}
