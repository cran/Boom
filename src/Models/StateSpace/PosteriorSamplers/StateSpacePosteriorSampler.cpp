/*
  Copyright (C) 2005-2011 Steven L. Scott

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

#include <Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp>
namespace BOOM{

  typedef StateSpacePosteriorSampler SSPS;
  typedef StateSpaceModelBase SSMB;

  SSPS::StateSpacePosteriorSampler(StateSpaceModelBase *model,
                                   RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        m_(model),
        latent_data_initialized_(false)
  {}

  void SSPS::draw(){
    if (!latent_data_initialized_) {
      m_->impute_state();
      latent_data_initialized_ = true;
    }
    impute_nonstate_latent_data();
    m_->observation_model()->sample_posterior();
    for(int s = 0; s < m_->nstate(); ++s) {
      m_->state_model(s)->sample_posterior();
    }
    m_->impute_state();
    // End with a call to impute_state() so that the internal state of
    // the Kalman filter matches up with the parameter draws.
  }

  double SSPS::logpri()const{
    double ans = m_->observation_model()->logpri();
    for(int s = 0; s < m_->nstate(); ++s){
      ans += m_->state_model(s)->logpri();
    }
    return ans;
  }

}  // namespace BOOM
