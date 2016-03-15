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

#include <cpputil/math_utils.hpp>
#include <cpputil/seq.hpp>
#include <cpputil/report_error.hpp>
#include <distributions.hpp>
#include <distributions/trun_gamma.hpp>
#include <Models/ChisqModel.hpp>
#include <Models/Glm/PosteriorSamplers/BregVsSampler.hpp>
#include <Models/MvnGivenScalarSigma.hpp>

namespace BOOM{

  typedef BregVsSampler BVS;

  namespace {
    Ptr<GammaModelBase> create_siginv_prior(RegressionModel *mod,
                                            double prior_nobs,
                                            double expected_rsq) {
      double sample_variance = mod->suf()->SST() / (mod->suf()->n() - 1);
      assert(expected_rsq > 0 && expected_rsq < 1);
      double sigma_guess = sqrt(sample_variance * (1-expected_rsq));
      return new ChisqModel(prior_nobs, sigma_guess);
    }
  }

  //----------------------------------------------------------------------
  BVS::BregVsSampler(RegressionModel * mod,
                     double prior_nobs,
                     double expected_rsq,
                     double expected_model_size,
                     bool first_term_is_intercept,
                     RNG &seeding_rng)
    : PosteriorSampler(seeding_rng),
      m_(mod),
      spri_(create_siginv_prior(mod, prior_nobs, expected_rsq)),
      indx(seq<uint>(0, m_->nvars_possible()-1)),
      max_nflips_(indx.size()),
      draw_beta_(true),
      draw_sigma_(true),
      // Initialize mutable workspace variables to illegal values.
      beta_tilde_(1, negative_infinity()),
      iV_tilde_(1, negative_infinity()),
      DF_(negative_infinity()),
      SS_(negative_infinity()),
      sigsq_sampler_(spri_)
  {
    uint p = m_->nvars_possible();
    Vector b = Vector(p, 0.0);
    if (first_term_is_intercept) {
      b[0] = m_->suf()->ybar();
    }
    SpdMatrix ominv(m_->suf()->xtx());
    double n = m_->suf()->n();
    ominv *= prior_nobs / n;
    bpri_ = new MvnGivenScalarSigma(ominv, mod->Sigsq_prm());

    double prob = expected_model_size/p;
    if (prob>1) prob = 1.0;
    Vector pi(p, prob);
    if (first_term_is_intercept) {
      pi[0] = 1.0;
    }

    vpri_ = new VariableSelectionPrior(pi);
    check_dimensions();
  }
  //----------------------------------------------------------------------
  BVS::BregVsSampler(RegressionModel *mod,
                     double prior_sigma_nobs,
                     double prior_sigma_guess,
                     double prior_beta_nobs,
                     double diagonal_shrinkage,
                     double prior_inclusion_probability,
                     bool force_intercept,
                     RNG &seeding_rng)
    : PosteriorSampler(seeding_rng),
      m_(mod),
      spri_(new ChisqModel(prior_sigma_nobs, prior_sigma_guess)),
      indx(seq<uint>(0, m_->nvars_possible()-1)),
      max_nflips_(indx.size()),
      draw_beta_(true),
      draw_sigma_(true),
      sigsq_sampler_(spri_)
  {
    uint p = m_->nvars_possible();
    Vector b = Vector(p, 0.0);
    double ybar = mod->suf()->ybar();
    b[0] = ybar;
    SpdMatrix ominv(m_->suf()->xtx());
    double n = m_->suf()->n();

    if (prior_sigma_guess <= 0) {
      ostringstream msg;
      msg << "illegal value of prior_sigma_guess in constructor "
          << "to BregVsSampler" << endl
          << "supplied value:  " << prior_sigma_guess << endl
          << "legal values are strictly > 0";
      report_error(msg.str());
    }
    ominv *= prior_beta_nobs/n;

    // handle diagonal shrinkage:  ominv =alpha*diag(ominv) + (1-alpha)*ominv
    // This prevents a perfectly singular ominv.
    double alpha = diagonal_shrinkage;
    if (alpha > 1.0 || alpha < 0.0) {
      ostringstream msg;
      msg << "illegal value of 'diagonal_shrinkage' in "
          << "BregVsSampler constructor.  Supplied value = "
          << alpha << ".  Legal values are [0, 1].";
      report_error(msg.str());
    }

    if (alpha < 1.0) {
      diag(ominv).axpy(diag(ominv), alpha/(1-alpha));
      ominv *= (1-alpha);
    }else{
      ominv.set_diag(diag(ominv));
    }

    bpri_ = new MvnGivenScalarSigma(b, ominv, m_->Sigsq_prm());

    Vector pi(p, prior_inclusion_probability);
    if (force_intercept) pi[0] = 1.0;

    vpri_ = new VariableSelectionPrior(pi);
    check_dimensions();
  }
  //----------------------------------------------------------------------
  BVS::BregVsSampler(RegressionModel *mod,
                     const Vector & b,
                     const SpdMatrix & Omega_inverse,
                     double sigma_guess,
                     double df,
                     const Vector &prior_inclusion_probs,
                     RNG &seeding_rng)
    : PosteriorSampler(seeding_rng),
      m_(mod),
      bpri_(new MvnGivenScalarSigma(b, Omega_inverse, m_->Sigsq_prm())),
      spri_(new ChisqModel(df, sigma_guess)),
      vpri_(new VariableSelectionPrior(prior_inclusion_probs)),
      indx(seq<uint>(0, m_->nvars_possible()-1)),
      max_nflips_(indx.size()),
      draw_beta_(true),
      draw_sigma_(true),
      sigsq_sampler_(spri_)
  {
    check_dimensions();
  }
  //----------------------------------------------------------------------
  BVS::BregVsSampler(RegressionModel *model,
                     const ZellnerPriorParameters &prior,
                     RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        m_(model),
        bpri_(new MvnGivenScalarSigma(
            prior.prior_beta_guess,
            prior.prior_beta_information,
            m_->Sigsq_prm())),
        spri_(new ChisqModel(prior.prior_sigma_guess_weight,
                             prior.prior_sigma_guess)),
        vpri_(new VariableSelectionPrior(prior.prior_inclusion_probabilities)),
        indx(seq<uint>(0, m_->nvars_possible()-1)),
        max_nflips_(indx.size()),
        draw_beta_(true),
        draw_sigma_(true),
        sigsq_sampler_(spri_)
  {
    check_dimensions();
  }
  //----------------------------------------------------------------------
  BVS::BregVsSampler(RegressionModel * mod,
                     Ptr<MvnGivenScalarSigmaBase> bpri,
                     Ptr<GammaModelBase> spri,
                     Ptr<VariableSelectionPrior> vpri,
                     RNG &seeding_rng)
    : PosteriorSampler(seeding_rng),
      m_(mod),
      bpri_(bpri),
      spri_(spri),
      vpri_(vpri),
      indx(seq<uint>(0, m_->nvars_possible()-1)),
      max_nflips_(indx.size()),
      draw_beta_(true),
      draw_sigma_(true),
      sigsq_sampler_(spri_)
  {
    check_dimensions();
  }
  //----------------------------------------------------------------------
  void BVS::limit_model_selection(uint n) { max_nflips_ =n;}
  void BVS::allow_model_selection(bool allow) {
    if (allow) {
      max_nflips_ = indx.size();
    } else suppress_model_selection();
  }
  void BVS::suppress_model_selection() {max_nflips_ =0;}
  void BVS::suppress_beta_draw() {draw_beta_ = false;}
  void BVS::allow_beta_draw() {draw_beta_ = false;}
  void BVS::suppress_sigma_draw() {draw_sigma_ = false;}
  void BVS::allow_sigma_draw() {draw_sigma_ = false;}

  //  since alpha = df/2 df is 2 * alpha, likewise for beta
  double BVS::prior_df() const { return 2 * spri_->alpha(); }
  double BVS::prior_ss() const { return 2 * spri_->beta(); }

  double BVS::log_model_prob(const Selector &g) const {
    if (g.nvars()==0) {
      // Integrate out sigma.  The empty model is handled as a special
      // case because information matrices cancel, and do not appear
      // in the sum of squares.  It is easier to handle them here than
      // to impose a global requirement about what logdet() should
      // mean for an empty matrix.
      double ss = m_->suf()->yty() + prior_ss();
      double df = m_->suf()->n() + prior_df();
      double ans = vpri_->logp(g) - (.5*df-1)*log(ss);
      return ans;
    }
    double ans = vpri_->logp(g);
    if (ans == negative_infinity()) {
      return ans;
    }
    double ldoi = set_reg_post_params(g, true);
    if (ldoi <= negative_infinity()) {
      return negative_infinity();
    }
    ans += .5*(ldoi - iV_tilde_.logdet());
    ans -= (.5*DF_-1)*log(SS_);
    return ans;
  }
  //----------------------------------------------------------------------
  double BVS::mcmc_one_flip(Selector &mod, uint which_var, double logp_old) {
    mod.flip(which_var);
    double logp_new = log_model_prob(mod);
    double u = runif (0,1);
    if (log(u) > logp_new - logp_old) {
      mod.flip(which_var);  // reject draw
      return logp_old;
    }
    return logp_new;
  }
  //----------------------------------------------------------------------
  void BVS::draw() {
    if (max_nflips_>0) draw_model_indicators();
    if (draw_beta_ || draw_sigma_) {
      set_reg_post_params(m_->coef().inc(),false);
    }
    if (draw_sigma_) draw_sigma();
    if (draw_beta_) draw_beta();
  }
  //----------------------------------------------------------------------
  bool BVS::model_is_empty() const {
    return m_->coef().inc().nvars()==0;
  }
  //----------------------------------------------------------------------
  void BVS::set_sigma_upper_limit(double sigma_upper_limit) {
    sigsq_sampler_.set_sigma_max(sigma_upper_limit);
  }

  void BVS::find_posterior_mode(double) {
    set_reg_post_params(m_->coef().inc(), true);
    m_->set_included_coefficients(beta_tilde_);
    m_->set_sigsq(SS_ / DF_);
  }

  //----------------------------------------------------------------------
  void BVS::draw_sigma() {
    double df, ss;
    if (model_is_empty()) {
      ss = m_->suf()->yty();
      df = m_->suf()->n();
    } else {
      df = DF_ - prior_df();
      ss = SS_ - prior_ss();
    }
    double sigsq = draw_sigsq_given_sufficient_statistics(df, ss);
    m_->set_sigsq(sigsq);
  }
  //----------------------------------------------------------------------
  void BVS::draw_beta() {
    if (model_is_empty()) return;
    iV_tilde_ /= m_->sigsq();
    beta_tilde_ = rmvn_ivar(beta_tilde_, iV_tilde_);
    m_->set_included_coefficients(beta_tilde_);
  }
  //----------------------------------------------------------------------
  void BVS::draw_model_indicators() {
    Selector g = m_->coef().inc();
    std::random_shuffle(indx.begin(), indx.end());
    double logp = log_model_prob(g);

    if (!std::isfinite(logp)) {
      ostringstream err;
      err << "BregVsSampler did not start with a legal configuration." << endl
          << "Selector vector:  " << g << endl
          << "beta: " << m_->included_coefficients() << endl;
      report_error(err.str());
    }

    uint n = std::min<uint>(max_nflips_, g.nvars_possible());
    for (uint i=0; i<n; ++i) {
      logp = mcmc_one_flip(g, indx[i], logp);
    }
    m_->coef().set_inc(g);
  }
  //----------------------------------------------------------------------
  double BVS::logpri() const {
    const Selector &g(m_->coef().inc());
    double ans = vpri_->logp(g);  // p(gamma)
    if (ans <= BOOM::negative_infinity()) return ans;

    double sigsq = m_->sigsq();
    ans += spri_->logp(1.0/sigsq);               // p(1/sigsq)

    if (g.nvars() > 0) {
      ans += dmvn(g.select(m_->Beta()),
                  g.select(bpri_->mu()),
                  g.select(bpri_->siginv()), true);
    }
    return ans;
  }
  //----------------------------------------------------------------------
  double BVS::set_reg_post_params(const Selector &g, bool do_ldoi) const {
    if (g.nvars()==0) {
      return 0;
    }
    Vector b = g.select(bpri_->mu());
    // Sigma = sigsq * Omega, so
    // siginv = ominv / sigsq, so
    // ominv = siginv * sigsq.
    SpdMatrix Ominv = g.select(bpri_->siginv()) * m_->sigsq();
    double ldoi = do_ldoi ? Ominv.logdet() : 0.0;

    Ptr<RegSuf> s = m_->suf();

    SpdMatrix xtx = s->xtx(g);
    Vector xty = s->xty(g);

    // iV_tilde_ / sigsq is the inverse of the posterior precision
    // matrix, given g.
    iV_tilde_ = Ominv + xtx;
    // beta_tilde_ is the posterior mean, given g
    beta_tilde_ = Ominv * b + xty;
    bool positive_definite = true;
    beta_tilde_ = iV_tilde_.solve(beta_tilde_, positive_definite);
    if (!positive_definite) {
      beta_tilde_ = Vector(iV_tilde_.nrow());
      return negative_infinity();
    }
    DF_ = s->n() + prior_df();
    SS_ = prior_ss();

    // Add in the sum of squared errors around beta_tilde_
    double likelihood_ss = s->yty() - 2*beta_tilde_.dot(xty)
        + xtx.Mdist(beta_tilde_);
    SS_ +=likelihood_ss;

    // Add in the sum of squares from the prior
    double prior_ss = Ominv.Mdist(beta_tilde_, b);
    SS_ += prior_ss;

    return ldoi;
  }

  void BVS::check_dimensions() const {
    if (vpri_->potential_nvars() != bpri_->dim()) {
      ostringstream err;
      err << "Objects of incompatible dimension were passed to the "
          << "BregVsSampler constructor." << endl
          << "The prior on the set of coefficients had dimension "
          << bpri_->dim()
          << ", while the prior on the set of inclusion indicators "
          << "had dimension "
          << vpri_->potential_nvars() << "."  << endl;
      report_error(err.str());
    }
  }
}  // namespace BOOM
