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

#include <Models/Glm/PosteriorSamplers/MlogitRwm.hpp>
#include <Models/MvnModel.hpp>
#include <distributions.hpp>

namespace BOOM{

  typedef MultinomialLogitModel MLM;
  typedef MlogitRwm MLR;

  MLR::MlogitRwm(MLM *mlm, Ptr<MvnBase> pri)
    : mlm_(mlm),
      pri_(pri)
  {}


  MLR::MlogitRwm(MLM *mlm,
		 const Vec &mu,
		 const Spd & Ominv)
    : mlm_(mlm),
      pri_(new MvnModel(mu, Ominv, true))
  {}



  void MLR::draw(){

    // random walk metropolis centered on current beta, with inverse
    // variance matrix given by current hessian of log posterior

    const Selector &inc(mlm_->coef().inc());
    uint p = inc.nvars();
    H.resize(p);
    g.resize(p);
    b = inc.select(mlm_->beta());
    mu = inc.select(pri_->mu());
    ivar = inc.select(pri_->siginv());

    double logp_old = mlm_->Loglike(g,H,2); // + dmvn(b,mu,ivar,0,true);

    H*= -1;
    H += ivar;  // now H is inverse posterior variance
    bstar = rmvt_ivar(b, H, 3);
    Spd Sigma = H.inv();
    //    cout << "sd's:  " << sqrt(Sigma.diag()) <<endl;
    mlm_->set_beta(inc.expand(bstar));

    double logp_new = mlm_->loglike(); // + dmvn(bstar,mu,ivar,0, true);

    double log_alpha = logp_new - logp_old;
    //    cout << logp_old << " " << logp_new << " ";

    double logu = log(runif(0,1));
    while(!std::isfinite(logu)) logu = log(runif(0,1));
    if(logu > log_alpha){  // reject the draw
      mlm_->set_beta(inc.expand(b));
      //      cout << "reject" << endl;
    }else{
      //      cout << "accept" << endl;
    }// otherwise accept, in which case nothing needs to be done.
  }


  double MLR::logpri()const{
    const Selector &inc(mlm_->coef().inc());
    Vector b = mlm_->coef().included_coefficients();
    return dmvn(b,
                inc.select(pri_->mu()),
                inc.select(pri_->siginv()),
                true);
  }
}
