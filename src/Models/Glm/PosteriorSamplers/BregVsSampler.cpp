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
#include <Models/Glm/PosteriorSamplers/BregVsSampler.hpp>
#include <Models/MvnGivenScalarSigma.hpp>

namespace BOOM{

  typedef BregVsSampler BVS;
  //----------------------------------------------------------------------
  BVS::BregVsSampler(RegressionModel * mod,
		     double prior_nobs,
		     double expected_rsq,
		     double expected_model_size,
                     bool first_term_is_intercept)
    : m_(mod),
      indx(seq<uint>(0, m_->nvars_possible()-1)),
      max_nflips_(indx.size()),
      draw_beta_(true),
      draw_sigma_(true),
      // Initialize mutable workspace variables to illegal values.
      beta_tilde_(1, negative_infinity()),
      iV_tilde_(1, negative_infinity()),
      DF_(negative_infinity()),
      SS_(negative_infinity())
  {
    uint p = m_->nvars_possible();
    Vec b = Vec(p, 0.0);
    if(first_term_is_intercept){
      b[0] = m_->suf()->ybar();
    }
    Spd ominv(m_->suf()->xtx());
    double n = m_->suf()->n();
    ominv *= prior_nobs/n;

    bpri_ = new MvnGivenScalarSigma(ominv, mod->Sigsq_prm());

    double v = m_->suf()->SST()/(n-1);
    assert(expected_rsq > 0 && expected_rsq < 1);
    double sigma_guess = sqrt(v * (1-expected_rsq));

    spri_ = new GammaModel(prior_nobs, pow(sigma_guess, 2)*prior_nobs/2);

    double prob = expected_model_size/p;
    if(prob>1) prob = 1.0;
    Vec pi(p, prob);
    if(first_term_is_intercept){
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
                     bool force_intercept)
    : m_(mod),
      indx(seq<uint>(0, m_->nvars_possible()-1)),
      max_nflips_(indx.size()),
      draw_beta_(true),
      draw_sigma_(true)
  {
    uint p = m_->nvars_possible();
    Vec b = Vec(p, 0.0);
    double ybar = mod->suf()->ybar();
    b[0] = ybar;
    Spd ominv(m_->suf()->xtx());
    double n = m_->suf()->n();

    if(prior_sigma_guess <= 0){
      ostringstream msg;
      msg << "illegal value of prior_sigma_guess in constructor to BregVsSampler"
          << endl
          << "supplied value:  " << prior_sigma_guess << endl
          << "legal values are strictly > 0";
      report_error(msg.str());
    }
    ominv *= prior_beta_nobs/n;

    // handle diagonal shrinkage:  ominv =alpha*diag(ominv) + (1-alpha)*ominv
    // This prevents a perfectly singular ominv.
    double alpha = diagonal_shrinkage;
    if(alpha > 1.0 || alpha < 0.0){
      ostringstream msg;
      msg << "illegal value of 'diagonal_shrinkage' in "
          << "BregVsSampler constructor.  Supplied value = "
          << alpha << ".  Legal values are [0, 1].";
      report_error(msg.str());
    }

    if(alpha < 1.0){
      diag(ominv).axpy(diag(ominv), alpha/(1-alpha));
      ominv *= (1-alpha);
    }else{
      ominv.set_diag(diag(ominv));
    }

    bpri_ = new MvnGivenScalarSigma(b, ominv, m_->Sigsq_prm());

    double prior_ss = pow(prior_sigma_guess, 2)*prior_sigma_nobs;
    spri_ = new GammaModel(prior_sigma_nobs/2, prior_ss/2);

    Vec pi(p, prior_inclusion_probability);
    if(force_intercept) pi[0] = 1.0;

    vpri_ = new VariableSelectionPrior(pi);
    check_dimensions();
  }
  //----------------------------------------------------------------------
  BVS::BregVsSampler(RegressionModel *mod,
                     const Vec & b,
                     const Spd & Omega_inverse,
                     double sigma_guess,
                     double df,
                     const Vec &prior_inclusion_probs)
    : m_(mod),
      bpri_(new MvnGivenScalarSigma(b, Omega_inverse, m_->Sigsq_prm())),
      spri_(new GammaModel(df/2, pow(sigma_guess, 2)*df/2)),
      vpri_(new VariableSelectionPrior(prior_inclusion_probs)),
      indx(seq<uint>(0, m_->nvars_possible()-1)),
      max_nflips_(indx.size()),
      draw_beta_(true),
      draw_sigma_(true)
  {
    check_dimensions();
  }
  //----------------------------------------------------------------------
  BVS::BregVsSampler(RegressionModel *model,
                     const ZellnerPriorParameters &prior)
      : m_(model),
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
        draw_sigma_(true)
  {
    check_dimensions();
  }
  //----------------------------------------------------------------------
  BVS::BregVsSampler(RegressionModel * mod,
		     Ptr<MvnGivenScalarSigma> bpri,
		     Ptr<GammaModelBase> spri,
		     Ptr<VariableSelectionPrior> vpri)
    : m_(mod),
      bpri_(bpri),
      spri_(spri),
      vpri_(vpri),
      indx(seq<uint>(0, m_->nvars_possible()-1)),
      max_nflips_(indx.size()),
      draw_beta_(true),
      draw_sigma_(true)
  {
    check_dimensions();
  }
  //----------------------------------------------------------------------
  void BVS::limit_model_selection(uint n){ max_nflips_ =n;}
  void BVS::allow_model_selection(){ max_nflips_ = indx.size();}
  void BVS::supress_model_selection(){max_nflips_ =0;}
  void BVS::supress_beta_draw(){draw_beta_ = false;}
  void BVS::allow_beta_draw(){draw_beta_ = false;}
  void BVS::supress_sigma_draw(){draw_sigma_ = false;}
  void BVS::allow_sigma_draw(){draw_sigma_ = false;}

  //  since alpha = df/2 df is 2 * alpha, likewise for beta
  double BVS::prior_df()const{ return 2 * spri_->alpha(); }
  double BVS::prior_ss()const{ return 2 * spri_->beta(); }

  double BVS::log_model_prob(const Selector &g)const{
    //    if(g.nvars()==0) return BOOM::negative_infinity();
    if(g.nvars()==0){
      // integrate out sigma
      double ss = m_->suf()->yty() + prior_ss();
      double df = m_->suf()->n() + prior_df();
      double ans = vpri_->logp(g) - (.5*df-1)*log(ss);
      return ans;
    }
    double ans = vpri_->logp(g);
    if(ans == negative_infinity()){
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
  double BVS::mcmc_one_flip(Selector &mod, uint which_var, double logp_old){
    mod.flip(which_var);
    double logp_new = log_model_prob(mod);
    double u = runif(0,1);
    if(log(u) > logp_new - logp_old){
      mod.flip(which_var);  // reject draw
      return logp_old;
    }
    return logp_new;
  }
  //----------------------------------------------------------------------
  void BVS::draw(){
    if(max_nflips_>0) draw_model_indicators();
    if(draw_beta_ || draw_sigma_){
      set_reg_post_params(m_->coef().inc(),false);
    }
    if(draw_sigma_) draw_sigma();
    if(draw_beta_) draw_beta();
  }
  //----------------------------------------------------------------------
  bool BVS::model_is_empty()const{
    return m_->coef().inc().nvars()==0;
  }
  //----------------------------------------------------------------------
  void BVS::draw_sigma(){
    double siginv = 0;
    if(model_is_empty()){
      double ss = m_->suf()->yty() + prior_ss();
      double df = m_->suf()->n() + prior_df();
      siginv = rgamma(df/2.0, ss/2.0);
    }else{
      siginv = rgamma(DF_/2.0, SS_/2.0);
    }
    m_->set_sigsq(1.0/siginv);
  }
  //----------------------------------------------------------------------
  void BVS::draw_beta(){
    if(model_is_empty()) return;
    iV_tilde_ /= m_->sigsq();
    beta_tilde_ = rmvn_ivar(beta_tilde_, iV_tilde_);
    m_->set_included_coefficients(beta_tilde_);
  }
  //----------------------------------------------------------------------
  void BVS::draw_model_indicators(){
    Selector g = m_->coef().inc();
    std::random_shuffle(indx.begin(), indx.end());
    double logp = log_model_prob(g);

    if(!std::isfinite(logp)){
      ostringstream err;
      err << "BregVsSampler did not start with a legal configuration." << endl
	  << "Selector vector:  " << g << endl
	  << "beta: " << m_->included_coefficients() << endl;
      report_error(err.str());
    }

    uint n = std::min<uint>(max_nflips_, g.nvars_possible());
    for(uint i=0; i<n; ++i){
      logp = mcmc_one_flip(g, indx[i], logp);
    }
    m_->coef().set_inc(g);
  }
  //----------------------------------------------------------------------
  double BVS::logpri()const{
    const Selector &g(m_->coef().inc());
    double ans = vpri_->logp(g);  // p(gamma)
    if(ans <= BOOM::negative_infinity()) return ans;

    double sigsq = m_->sigsq();
    ans += spri_->logp(1.0/sigsq);               // p(1/sigsq)

    if(g.nvars() > 0){
      ans += dmvn(g.select(m_->Beta()),
                  g.select(bpri_->mu()),
                  g.select(bpri_->siginv()), true);
    }
    return ans;
  }
  //----------------------------------------------------------------------
  double BVS::set_reg_post_params(const Selector &g, bool do_ldoi)const{
    if(g.nvars()==0){
      return 0;
    }
    Vec b = g.select(bpri_->mu());
    Spd Ominv = g.select(bpri_->ominv());
    double ldoi = do_ldoi ? Ominv.logdet() : 0.0;

    Ptr<RegSuf> s = m_->suf();

    Spd xtx = s->xtx(g);
    Vec xty = s->xty(g);

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

  void BVS::check_dimensions()const{
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
}
