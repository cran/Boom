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

#ifndef BOOM_CONJUGATE_PRIOR_POLICY_HPP
#define BOOM_CONJUGATE_PRIOR_POLICY_HPP
#include <Models/Policies/PriorPolicy.hpp>
#include <cpputil/report_error.hpp>

namespace BOOM{
  template <class CONJ>
  class ConjugatePriorPolicy
    : public PriorPolicy
  {
  public:
    typedef ConjugatePriorPolicy<CONJ> ConjPriorPolicy;

    ConjugatePriorPolicy * clone()const=0;
    //    virtual void set_method(Ptr<PosteriorSampler>);
    virtual void set_conjugate_prior(Ptr<CONJ>);
    virtual void clear_methods();
    Ptr<CONJ> get_conjugate_prior()const;

    virtual void find_posterior_mode();
  private:
    Ptr<CONJ> c_;
  };

//   template <class CONJ>
//   void ConjugatePriorPolicy<CONJ>::set_method(Ptr<PosteriorSampler> p){
//     Ptr<CONJ> c = p.dcast<CONJ>();
//     if(!c) PriorPolicy::set_method(p);
//     else set_conjugate_prior(c);
//   }

  template <class CONJ>
  void ConjugatePriorPolicy<CONJ>::set_conjugate_prior(Ptr<CONJ> c){
    c_ = c;
    PriorPolicy::set_method(c);
  }

  template <class CONJ>
  void ConjugatePriorPolicy<CONJ>::clear_methods(){
    PriorPolicy::clear_methods();
    c_.reset();
  }

  template <class CONJ>
  Ptr<CONJ> ConjugatePriorPolicy<CONJ>::get_conjugate_prior()const{
    if(!!c_) return c_;
    ostringstream err;
    err << "conjugate prior has not been set" << endl
	<< typeid(*c_).name() << endl;
    report_error(err.str());
    return c_;
  }

  template <class CONJ>
  void ConjugatePriorPolicy<CONJ>::find_posterior_mode(){
    Ptr<CONJ> pri = get_conjugate_prior();
    pri->find_posterior_mode();
  }
}

#endif// BOOM_CONJUGATE_PRIOR_POLICY_HPP
