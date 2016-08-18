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


#include <cpputil/RefCounted.hpp>
#include <cpputil/Ptr.hpp>
#include <LinAlg/Vector.hpp>
#include <distributions/rng.hpp>

namespace BOOM{

  // The job of a PosteriorSampler is primarily to simulate a set of
  // model parameters from their posterior distribution.  Concrete
  // instances of a PosteriorSampler should contain a "dumb" pointer
  // to a specific concrete Model to be managed (because the model
  // owns the Sampler, not the other way around), as well as a Ptr to
  // one or more other model objects constituting the prior.
  //
  // Some PosteriorSamplers also allow you to find the posterior mode
  // of the model that they manage.  If so, they should override the
  // can_find_posterior_mode method to return true.
  class PosteriorSampler
    : private RefCounted {
  public:
    PosteriorSampler(RNG &seeding_rng);
    PosteriorSampler(const PosteriorSampler &);
    virtual void draw() = 0;
    virtual double logpri() const = 0;
    ~PosteriorSampler() override{}
    RNG & rng()const{return rng_;}
    void set_seed(unsigned long);

    // Returns true if the child class implements
    // find_posterior_mode().  Returns false otherwise.
    virtual bool can_find_posterior_mode() const {
      return false;
    }

    // The default implementation of this function throws an exception
    // through report_error().
    virtual void find_posterior_mode(double epsilon = 1e-5);

    friend void intrusive_ptr_add_ref(PosteriorSampler *m);
    friend void intrusive_ptr_release(PosteriorSampler *m);
   private:
    mutable RNG rng_;
  };

}  // namespace BOOM
#endif// BOOM_SAMPLING_METHOD_HPP
