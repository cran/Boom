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

#include <Models/CompositeEmMixtureComponent.hpp>
#include <cpputil/report_error.hpp>

namespace BOOM{

  typedef CompositeEmMixtureComponent CME;
  typedef CompositeModel CM;

  CME::CompositeEmMixtureComponent() {}

  CME::CompositeEmMixtureComponent(const CME &rhs)
    : Model(rhs),
      CompositeModel(rhs),
      EmMixtureComponent(rhs)
  {
    int S = components().size();
    em_components_.reserve(S);
    for (int s = 0; s < S; ++s) {
      em_components_.push_back(components()[s].dcast<EmMixtureComponent>());
    }
  }

  CME * CME::clone()const{return new CME(*this);}

  void CME::mle(){
    for(int s = 0; s < em_components_.size(); ++s){
      em_components_[s]->mle();
    }
  }

  void CME::find_posterior_mode(){
    for(uint s=0; s<em_components_.size(); ++s){
      em_components_[s]->find_posterior_mode();
    }
  }

  void CME::add_mixture_data(Ptr<Data> dp, double prob){
    Ptr<CompositeData> d(CM::DAT(dp));
    uint S = em_components_.size();
    assert(d->dim() == S);
    for(uint s=0; s<S; ++s){
      em_components_[s]->add_mixture_data(d->get_ptr(s), prob);
    }
  }

  void CME::add_model(Ptr<MixtureComponent> new_model){
    Ptr<EmMixtureComponent> em_model = new_model.dcast<EmMixtureComponent>();
    if (!em_model) {
      report_error("Could not convert argument to an EmMixtureComponent in "
                   "CompositeEmMixtureComponent::"
                   "add_model(Ptr<MixtureComponent>)");
    }
    em_components_.push_back(em_model);
    CM::add_model(em_model);
  }

  // void CME::add_model(Ptr<EmMixtureComponent> new_model){
  //   em_components_.push_back(new_model);
  //   CM::add_model(new_model);
  // }
}
