#ifndef BOOM_GLM_SPIKE_SLAB_SAMPLER_HPP_
#define BOOM_GLM_SPIKE_SLAB_SAMPLER_HPP_

#include <Models/Glm/Glm.hpp>
#include <Models/MvnBase.hpp>
#include <Models/Glm/VariableSelectionPrior.hpp>
#include <Models/Glm/WeightedRegressionModel.hpp>

namespace BOOM {

  // A class to manage the elements of spike-and-slab posterior
  // sampling common to GlmModel objects.  This class does not inherit
  // from PosteriorSampler because it is intended to be an element of
  // a model-specific PosteriorSampler class that can impute the
  // latent variables needed to turn the GLM into a Gaussian problem.
  class SpikeSlabSampler {
   public:
    SpikeSlabSampler(GlmModel *model,
                     Ptr<MvnBase> slab_prior,
                     Ptr<VariableSelectionPrior> spike_prior);
    double logpri() const;

    // Performs one MCMC sweep along the inclusion indicators for the
    // managed GlmModel.
    void draw_model_indicators(RNG &rng, const WeightedRegSuf &suf);

    // Draws the set of included Glm coefficients given complete data
    // sufficient statistics.
    void draw_beta(RNG &rng, const WeightedRegSuf &suf);

    // If tf == true then draw_model_indicators is a no-op.  Otherwise
    // model indicators will be sampled each iteration.
    void allow_model_selection(bool tf);

    // In very large problems you may not want to sample every element
    // of the inclusion vector each time.  If max_flips is set to a
    // positive number then at most that many randomly chosen
    // inclusion indicators will be sampled.
    void limit_model_selection(int max_flips);

   private:
    double log_model_prob(const Selector &g, const WeightedRegSuf &suf) const;
    double mcmc_one_flip(
        RNG &rng,
        Selector &g,
        int which_variable,
        double logp_old,
        const WeightedRegSuf &suf);

    GlmModel *model_;
    Ptr<MvnBase> slab_prior_;
    Ptr<VariableSelectionPrior> spike_prior_;
    int max_flips_;
    bool allow_model_selection_;
  };

}  // namespace BOOM


#endif // BOOM_GLM_SPIKE_SLAB_SAMPLER_HPP_
