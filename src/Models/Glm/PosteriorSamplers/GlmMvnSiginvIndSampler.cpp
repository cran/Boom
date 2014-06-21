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

#include <Models/Glm/PosteriorSamplers/GlmMvnSiginvIndSampler.hpp>
#include <Models/Glm/GlmMvnPrior.hpp>
#include <LinAlg/Vector.hpp>
#include <LinAlg/SpdMatrix.hpp>
#include <Models/GammaModel.hpp>
#include <distributions.hpp>

namespace BOOM{

  typedef GlmMvnSiginvIndSampler GSI;

  GSI::GlmMvnSiginvIndSampler(GlmMvnPrior *mod, Gvec ivar_ptr)
    : mod_(mod),
      ivar_pri_(ivar_ptr)
  {}

  void GSI::draw(){

    Spd sumsq = mod_->sumsq();
    const VectorView ss_diag(sumsq.diag());
    const Vec & nobs(mod_->suf()->vnobs());
    uint K = dim();
    Vec ivar(K);

    for(uint k=0; k<K; ++k){
      double alpha = ivar_pri_[k]->alpha();
      double df = nobs[k];
      double beta = ivar_pri_[k]->beta();
      double ss = ss_diag[k];
      ivar[k] = rgamma(alpha + df/2, beta + ss/2);
    }
    Spd Siginv(K);
    Siginv.set_diag(ivar);
    mod_->set_siginv(Siginv);
  }

  double GSI::logpri()const{
    const Spd & siginv(mod_->siginv());
    ConstVectorView d(siginv.diag());
    double ans=0;
    uint K = dim();
    for(uint k=0; k<K; ++k){
      double a = ivar_pri_[k]->alpha();
      double b = ivar_pri_[k]->beta();
      ans += dgamma(d[k], a,b, true);
    }
    return ans;
  }

  uint GSI::dim()const{ return mod_->dim(); }
}
