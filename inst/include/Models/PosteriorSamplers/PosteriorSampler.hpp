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

#ifndef BOOM_SAMPLING_METHOD_HPP
#define BOOM_SAMPLING_METHOD_HPP

#include <LinAlg/Types.hpp>
#include <cpputil/RefCounted.hpp>
#include <cpputil/Ptr.hpp>
#include <LinAlg/Vector.hpp>
#include <distributions/rng.hpp>

namespace BOOM{

  class Params;

  class PosteriorSampler
    : private RefCounted{
  public:
    PosteriorSampler();
    PosteriorSampler(const PosteriorSampler &);
    virtual void draw()=0;
    virtual double logpri()const=0;
    virtual ~PosteriorSampler(){}
    friend void intrusive_ptr_add_ref(PosteriorSampler *m);
    friend void intrusive_ptr_release(PosteriorSampler *m);
    RNG & rng()const{return rng_;}
    void set_seed(unsigned long);
   private:
    mutable RNG rng_;
  };

  void intrusive_ptr_add_ref(PosteriorSampler *m);
  void intrusive_ptr_release(PosteriorSampler *m);
}
#endif// BOOM_SAMPLING_METHOD_HPP
