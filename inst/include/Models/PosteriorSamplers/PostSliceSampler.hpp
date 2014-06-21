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
#ifndef BOOM_POST_SLICE_SAMPLER_HPP
#define BOOM_POST_SLICE_SAMPLER_HPP
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/ParamTypes.hpp>
#include <TargetFun/LoglikeSubset.hpp>
#include <boost/function.hpp>

namespace BOOM{
  class LoglikeSubsetTF;
  class LogPostTF;
  class SliceSampler;
  class LoglikeModel;
  class VectorModel;

  class PosteriorSliceSampler : virtual public PosteriorSampler{
  public:
    PosteriorSliceSampler(Ptr<Params>,
			  LoglikeModel *,
			  Ptr<VectorModel>,
			  bool unimodal=false);

    PosteriorSliceSampler(const ParamVec &,
			  LoglikeModel *,
			  Ptr<VectorModel>,
			  bool unimodal=false);

    PosteriorSliceSampler(LoglikeModel *,
                          Ptr<VectorModel>,
                          bool unimodal=false);
    virtual void draw();
    virtual double logpri()const;
  private:
    ParamVec prms;
    LoglikeSubsetTF loglike;
    Ptr<VectorModel> pri;
    boost::function<double(const Vec &)> logpost;
    mutable Vec x;   // temporary workspace
    Ptr<SliceSampler> sampler;
  };

}
#endif// BOOM_POST_SLICE_SAMPLER_HPP
