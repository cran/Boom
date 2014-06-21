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
#ifndef BOOM_SAMPLERS_HPP
#define BOOM_SAMPLERS_HPP
#include <cpputil/RefCounted.hpp>
#include <LinAlg/Types.hpp>
#include <distributions/rng.hpp>

namespace BOOM{

  /*============================================================

    A Sampler is an object for drawing from a generic target
    distribution.  Samplers are used to implement PosteriorSamplers
    used for drawing from posterior distributions.  See
    Models/PosteriorSamplers.

    ============================================================*/

  class SamplerBase : private RefCounted{
  public:
    SamplerBase();
    SamplerBase(const SamplerBase &rhs);
    virtual ~SamplerBase();
    void set_seed(unsigned long s);
    RNG & rng()const;
    friend void intrusive_ptr_add_ref(SamplerBase *s){s->up_count();}
    friend void intrusive_ptr_release(SamplerBase *s){
      s->down_count(); if(s->ref_count()==0) delete s;}
    void set_rng(RNG *r, bool owns_rng=true);
   private:
    mutable RNG *rng_;
    bool owns_rng_;
  };

  class Sampler : public SamplerBase{
  public:
    virtual Vec draw(const Vec &old)=0;
    virtual double logp(const Vec &x)const=0;
    // logp evaluates the log pdf of the density being sampled
  };
  //======================================================================

  class ScalarSampler : public SamplerBase{
  public:
    virtual double draw(double)=0;
    virtual double logp(double)const=0;
    // logp evaluates the log pdf of the density being sampled
  };

}
#endif // BOOM_SAMPLERS_HPP
