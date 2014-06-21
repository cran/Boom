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

#include <Models/Glm/PosteriorSamplers/GlmMvnPriorMeanSampler.hpp>
#include <distributions.hpp>

namespace BOOM{
  typedef GlmMvnPriorMeanSampler GMS;

  GMS::GlmMvnPriorMeanSampler(GlmMvnPrior *Mod, Ptr<MvnBase> Pri)
    : mod(Mod),
      pri(Pri),
      ivar(Mod->dim()),
      mu(Mod->dim())
  {}

  void GMS::draw(){
    ivar = pri->siginv();
    mu = ivar * pri->mu();
    const Spd & siginv(mod->siginv());
    Ptr<GlmMvnSuf> s(mod->suf());
    ivar+= el_mult(s->GTG(), siginv);
    uint K = mu.size();
    const Mat & btg(s->BTG());
    for(uint k=0; k<K; ++k)
      mu[k]+= siginv.col(k).dot(btg.col(k));

    // at this point we fail if any elements of ivar are 0.
    // solve the problem by dropping those coefficients.

    Selector inc(K, true);
    const VectorView ivar_diag(ivar.diag());
    for(uint k=0; k<K; ++k){
      if(ivar_diag[k]==0) inc.drop(k);
    }
    ivar = inc.select(ivar);
    mu = inc.select(mu);
    mu = inc.expand(rmvn_suf(ivar, mu));
    mod->set_mu(mu);
  }

  double GMS::logpri()const{ return pri->logp(mod->mu()); }

}
