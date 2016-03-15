/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#ifndef BOOM_BINOMIAL_LOGIT_SPIKE_SLAB_SAMPLER_HPP_
#define BOOM_BINOMIAL_LOGIT_SPIKE_SLAB_SAMPLER_HPP_

#include <Models/Glm/PosteriorSamplers/BinomialLogitAuxmixSampler.hpp>
#include <Models/Glm/VariableSelectionPrior.hpp>
#include <LinAlg/Selector.hpp>

namespace BOOM{
  class BinomialLogitSpikeSlabSampler : public BinomialLogitAuxmixSampler{
   public:
    BinomialLogitSpikeSlabSampler(BinomialLogitModel *m,
                                  Ptr<MvnBase> pri,
                                  Ptr<VariableSelectionPrior> vpri,
                                  int clt_threshold,
                                  RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    void draw_model_indicators();
    virtual void draw_beta();
    double log_model_prob(const Selector &gamma)const;

    // toggles whether or not draw_model_indicators is called as part
    // of draw().  If you don't want model selection to be part of the
    // MCMC then you need to manually manage the include/exclude
    // decisions through m_->coef(), preferably before the model is
    // passed to the constructor
    void allow_model_selection(bool tf);

    // If max_flips > 0 then at most max_flips variable inclusion
    // indicators will be sampled each iteration.  Samples occur in
    // random order.  If max_flips <= 0 then all available variables
    // will be sampled.
    void limit_model_selection(int max_flips);

    // Locate and set model paramters to the posterior mode of the
    // included variables, given inclusion.  Save the un-normalized
    // log posterior (the objective function) in posterior_mode_value_;.
    void find_posterior_mode(double epsilon = 1e-5) override;

    bool can_find_posterior_mode() const override {
      return true;
    }

    bool posterior_mode_found() const {
      return posterior_mode_found_;
    }

    double log_posterior_at_mode() const {
      return log_posterior_at_mode_;
    }

   private:
    double mcmc_one_flip(Selector &mod, uint which_var, double logp_old);
    BinomialLogitModel *m_;
    Ptr<MvnBase> pri_;
    Ptr<VariableSelectionPrior> vpri_;
    bool allow_model_selection_;
    int max_flips_;
    bool posterior_mode_found_;
    double log_posterior_at_mode_;
  };

}  // namespace BOOM
#endif // BOOM_BINOMIAL_LOGIT_SPIKE_SLAB_SAMPLER_HPP_
