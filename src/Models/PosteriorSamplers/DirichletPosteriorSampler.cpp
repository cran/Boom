/*
  Copyright (C) 2005-2009 Steven L. Scott

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
#include <Models/PosteriorSamplers/DirichletPosteriorSampler.hpp>
#include <distributions.hpp>
#include <Samplers/ScalarSliceSampler.hpp>
#include <cpputil/math_utils.hpp>

namespace BOOM{
  typedef DirichletPosteriorSampler DPS;

  DPS::DirichletPosteriorSampler(DirichletModel *mod,
                                 Ptr<VectorModel> Phi,
                                 Ptr<DoubleModel> alpha)
    : mod_(mod),
      phi_prior_(Phi),
      alpha_prior_(alpha)
  {}
//----------------------------------------------------------------------
  void DPS::draw(){
    Vec nu = mod_->nu();
    uint d=nu.size();
    for(uint i=0; i<d; ++i){
      DirichletLogp logp(i, nu,
			 mod_->suf()->sumlog(), mod_->suf()->n(),
			 phi_prior_, alpha_prior_);
      ScalarSliceSampler sam(logp, true);
      sam.set_lower_limit(0);
      nu[i] = sam.draw(nu[i]);
    }
    mod_->set_nu(nu);
  }

//----------------------------------------------------------------------
double DPS::logpri()const{
  const Vec & nu(mod_->nu());
  double alpha = sum(nu);
  double ans = alpha_prior_->logp(alpha);
  ans+= phi_prior_->logp(nu/alpha);
  // Add in the Jacobian term to make the prior with respect to nu.
  ans -= (dim()-1) * log(alpha);
  return ans;
}
//----------------------------------------------------------------------
uint DPS::dim()const{ return mod_->nu().size(); }

//----------------------------------------------------------------------
  typedef DirichletLogp DLP;
  DLP::DirichletLogp(uint pos,  const Vec & nu, const Vec & sumlogpi,
		     double nobs, Ptr<VectorModel> phi, Ptr<DoubleModel> alpha,
                     double min_nu)
    : sumlogpi_(sumlogpi),
      nobs_(nobs),
      pos_(pos),
      nu_(nu),
      min_nu_(min_nu),
      alpha_prior_(alpha),
      phi_prior_(phi)
  {}
//----------------------------------------------------------------------
  double DLP::operator()(double nu)const{
    if(nu < min_nu_) return BOOM::negative_infinity();
    nu_[pos_]=nu;
    return logp();
  }
//----------------------------------------------------------------------
  double DLP::logp()const{
    double alpha = sum(nu_);
    if(alpha<=0) return BOOM::negative_infinity();
    uint d = nu_.size();
    double ans = alpha_prior_->logp(alpha);   // alpha prior
    if(!std::isfinite(ans)) return ans;
    ans+= phi_prior_->logp(nu_/alpha);        // phi prior
    if(!std::isfinite(ans)) return ans;
    ans-= (d-1) * log(alpha);                 // jacobian
    ans += dirichlet_loglike(nu_, 0,0,sumlogpi_, nobs_);
    return ans;
  }
//----------------------------------------------------------------------



}
