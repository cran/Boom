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

#ifndef BOOM_BINOMIAL_LOGIT_COMPOSITE_SPIKE_SLAB_SAMPLER_HPP_
#define BOOM_BINOMIAL_LOGIT_COMPOSITE_SPIKE_SLAB_SAMPLER_HPP_
#include <Models/Glm/PosteriorSamplers/BinomialLogitSpikeSlabSampler.hpp>
#include <Models/MvnBase.hpp>

namespace BOOM{
  //======================================================================
  // A functor that returns the log posterior and first two
  // derivatives for the specified chunk.
  class BinomialLogitLogPostChunk {
   public:
    BinomialLogitLogPostChunk(const BinomialLogitModel *model,
                              const MvnBase *prior,
                              int chunk_size,
                              int chunk_number)
        : m_(model),
          pri_(prior),
          start_(chunk_size * chunk_number)
    {
      int nvars = m_->coef().nvars();
      int elements_remaining = nvars - start_;
      chunk_size_ = std::min(chunk_size, elements_remaining);
    }
    double operator()(const Vec &beta_chunk)const;
    double operator()(const Vec &beta_chunk, Vec &grad, Mat &hess, int nd)const;
   private:
    const BinomialLogitModel *m_;
    const MvnBase * pri_;
    int start_;
    int chunk_size_;
  };

  //======================================================================
  // The BinomialLogitSpikeSlabSampler can be slow in the presence of
  // a separating hyperplane, but it is good at making
  // trans-dimensional moves.  One solution is to bundle the
  // spike-and-slab sampler with one or more fixed dimensional samplers.
  // At each iteration one of the samplers is chosen at random and used
  // to generate a move.  This class combines the AuxiliaryMixture
  // sampler with RandomWalkMetropolis and
  // TailoredIndependenceMetropolis (TIM) samplers.
  class BinomialLogitCompositeSpikeSlabSampler
      : public BinomialLogitSpikeSlabSampler {
   public:
    BinomialLogitCompositeSpikeSlabSampler(
        BinomialLogitModel *model,
        Ptr<MvnBase> prior,
        Ptr<VariableSelectionPrior> vpri,
        int clt_threshold,
        double tdf,
        int max_chunk_size,
        double rwm_variance_scale_factor = 1.0);
    virtual void draw();
    void rwm_draw();
    void tim_draw();

    // Draw the specified chunk using a random walk proposal.
    void rwm_draw_chunk(int chunk);

    BinomialLogitLogPostChunk log_posterior(int chunk)const;

    ostream & time_report(ostream &out)const;

   private:
    BinomialLogitModel *m_;
    Ptr<MvnBase> pri_;
    double tdf_;
    int max_chunk_size_;
    double rwm_variance_scale_factor_;

    int auxmix_tries_;
    double auxmix_times_;
    int rwm_tries_;
    double rwm_times_;
    int tim_tries_;
    double tim_times_;

    int rwm_chunk_attempts_;
    int rwm_chunk_successes_;

    int tim_chunk_attempts_;
    int tim_chunk_mode_failures_;
    int tim_chunk_successes_;

    double rwm_chunk_times_;
    double tim_mode_finding_times_;
    double tim_trial_times_;
    double tim_mode_finding_wasted_time_;

    // Compute the size of the largest chunk
    int compute_chunk_size()const;
    int compute_number_of_chunks()const;
  };
}
#endif //  BOOM_BINOMIAL_LOGIT_COMPOSITE_SPIKE_SLAB_SAMPLER_HPP_
