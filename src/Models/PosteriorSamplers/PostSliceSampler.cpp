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
#include <Models/PosteriorSamplers/PostSliceSampler.hpp>
#include <TargetFun/LoglikeSubset.hpp>
#include <Samplers/SliceSampler.hpp>
#include <Models/ParamTypes.hpp>
#include <boost/bind.hpp>
#include <Models/VectorModel.hpp>

namespace BOOM{

  namespace{
    struct Logp{
      Logp(LoglikeSubsetTF Loglike, Ptr<VectorModel> Pri)
	: loglike(Loglike), pri(Pri){}
      double operator()(const Vec &x)const{
	return loglike(x) + pri->logp(x); }

      LoglikeSubsetTF loglike;
      Ptr<VectorModel> pri;
    };
  }

  PosteriorSliceSampler::PosteriorSliceSampler
  (Ptr<Params> prm, LoglikeModel *mod, Ptr<VectorModel> Pri, bool unimodal)
    : prms(1, prm),
      loglike(mod, prms),
      pri(Pri),
      logpost(Logp(loglike, pri)),
      x(vectorize(prms)),
      sampler(new SliceSampler(logpost, unimodal))
  {}

  PosteriorSliceSampler::PosteriorSliceSampler
  (const ParamVec &Prms, LoglikeModel *mod, Ptr<VectorModel> Pri, bool unimodal)
    : prms(Prms),
      loglike(mod, prms),
      pri(Pri),
      logpost(Logp(loglike, pri)),
      x(vectorize(prms)),
      sampler(new SliceSampler(logpost, unimodal))
  {}

  PosteriorSliceSampler::PosteriorSliceSampler
  (LoglikeModel *mod, Ptr<VectorModel> Pri, bool unimodal)
    : prms(mod->t()),
      loglike(mod, prms),
      pri(Pri),
      logpost(Logp(loglike, pri)),
      x(vectorize(prms)),
      sampler(new SliceSampler(logpost, unimodal))
  {}

  void PosteriorSliceSampler::draw(){
    x = vectorize(prms);
    x = sampler->draw(x);
    unvectorize(prms,x);
  }

  double PosteriorSliceSampler::logpri()const{
    x = vectorize(prms);
    return pri->logp(x);
  }

}
