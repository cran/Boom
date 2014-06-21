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
#include <Samplers/MetropolisHastings.hpp>
#include <distributions.hpp>

namespace BOOM{
  typedef MetropolisHastings MH;

  MH::MetropolisHastings(const Target & target, Ptr<MH_Proposal> prop)
      : f_(target),
        prop_(prop),
        accepted_(false)
  {}

  void MH::set_proposal(Ptr<MH_Proposal> p){
    prop_ = p;
  }

  void MH::set_target(Target f){ f_ = f;}

  Vec MH::draw(const Vec & old){
    cand_ = prop_->draw(old);
    double num = logp(cand_) - logp(old);
    double denom = 0;
    if(!prop_->sym()){
      denom = prop_->logf(cand_,old) - prop_->logf(old,cand_);
    }
//      cout << "MH::draw()... cand - old = " << cand_ - old << endl
//           << "num   = " << num << endl
//           << "denom = " << denom << endl
//          ;

    double u = log(runif_mt(rng()));
    accepted_ = u < num - denom;
    return accepted_ ? cand_ : old;
  }

  bool MH::last_draw_was_accepted()const{
    return accepted_;
  }

  double MH::logp(const Vec &x)const{
    return f_(x);
  }


  typedef ScalarMetropolisHastings SMH;
  SMH::ScalarMetropolisHastings(const ScalarTarget &f,
                                Ptr<MH_ScalarProposal> prop)
      : f_(f),
        prop_(prop),
        accepted_(false)
  {}

  double SMH::draw(double old){
    double cand = prop_->draw(old);
    double num = f_(cand) - f_(old);
    double denom = 0;
    if(!prop_->sym()){
      denom = prop_->logf(cand,old) - prop_->logf(old,cand);
    }
    double u = log(runif_mt(rng()));
    accepted_ = u < num - denom;
    return accepted_ ? cand : old;
  }

  bool SMH::last_draw_was_accepted()const{
    return accepted_;
  }

  double SMH::logp(double x)const{ return f_(x);}
}
