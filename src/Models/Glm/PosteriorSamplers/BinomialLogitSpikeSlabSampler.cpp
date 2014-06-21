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
#include <Models/Glm/PosteriorSamplers/BinomialLogitSpikeSlabSampler.hpp>
#include <cpputil/seq.hpp>
#include <distributions.hpp>
#include <cpputil/math_utils.hpp>

namespace BOOM{
  typedef BinomialLogitSpikeSlabSampler BLSSS;

  BLSSS::BinomialLogitSpikeSlabSampler(BinomialLogitModel *m,
                                       Ptr<MvnBase> pri,
                                       Ptr<VariableSelectionPrior> vpri,
                                       int clt_threshold)
      : BinomialLogitAuxmixSampler(m, pri, clt_threshold),
        m_(m),
        pri_(pri),
        vpri_(vpri),
        allow_model_selection_(true),
        max_flips_(-1)
  {}

  void BLSSS::draw(){
    impute_latent_data();
    if(allow_model_selection_) draw_model_indicators();
    draw_beta();
  }

  void BLSSS::draw_beta(){
    Selector g = m_->coef().inc();
    if(g.nvars() == 0){
      m_->drop_all();
      return;
    }
    SpdMatrix ivar = g.select(pri_->siginv());
    Vector ivar_mu = ivar * g.select(pri_->mu());
    ivar += g.select(suf().xtx());
    ivar_mu += g.select(suf().xty());
    Vector b = ivar.solve(ivar_mu);
    b = rmvn_ivar(b, ivar);

    // If model selection is turned off and some elements of beta
    // happen to be zero (because, e.g., of a failed MH step) we don't
    // want the dimension of beta to change.
    m_->set_included_coefficients(b, g);
  }

  double BLSSS::logpri()const{
    const Selector & g(m_->coef().inc());
    double ans = vpri_->logp(g);  // p(gamma)
    if(ans == BOOM::negative_infinity()) return ans;
    if(g.nvars() > 0){
      ans += dmvn(m_->included_coefficients(),
                  g.select(pri_->mu()),
                  g.select(pri_->siginv()),
                  true);
    }
    return ans;
  }

  double BLSSS::log_model_prob(const Selector &g)const{
    // borrowed from MLVS.cpp
    double num = vpri_->logp(g);
    if(num==BOOM::negative_infinity() || g.nvars() == 0){
      // If num == -infinity then it is in a zero support point in the
      // prior.  If g.nvars()==0 then all coefficients are zero
      // because of the point mass.  The only entries remaining in the
      // likelihood are sums of squares of y[i] that are independent
      // of g.  They need to be omitted here because they are omitted
      // in the non-empty case below.
      return num;
    }
    SpdMatrix ivar = g.select(pri_->siginv());
    num += .5*ivar.logdet();
    if(num == BOOM::negative_infinity()) return num;

    Vector mu = g.select(pri_->mu());
    Vector ivar_mu = ivar * mu;
    num -= .5*mu.dot(ivar_mu);

    bool ok=true;
    ivar += g.select(suf().xtx());
    Mat L = ivar.chol(ok);
    if(!ok)  return BOOM::negative_infinity();
    double denom = sum(log(L.diag()));  // = .5 log |ivar|
    Vec S = g.select(suf().xty()) + ivar_mu;
    Lsolve_inplace(L,S);
    denom-= .5*S.normsq();  // S.normsq =  beta_tilde ^T V_tilde beta_tilde
    return num-denom;
  }

  void BLSSS::allow_model_selection(bool tf){
    allow_model_selection_ = tf;
  }

  void BLSSS::limit_model_selection(int max_flips){
    max_flips_ = max_flips;
  }

  void BLSSS::draw_model_indicators(){
    Selector g = m_->coef().inc();
    std::vector<int> indx = seq<int>(0, g.nvars_possible()-1);
    // I'd like to rely on std::random_shuffle for this, but I want
    // control over the random number generator.
    for (int i = 0; i < indx.size(); ++i) {
      int j = random_int_mt(rng(), 0, indx.size() - 1);
      std::swap(indx[i], indx[j]);
    }

    double logp = log_model_prob(g);

    if(!std::isfinite(logp)){
      vpri_->make_valid(g);
      logp = log_model_prob(g);
    }
    if(!std::isfinite(logp)){
      ostringstream err;
      err << "BinomialLogitSpikeSlabSampler did not start with a "
          << "legal configuration."
          << endl << "Selector vector:  " << g << endl
          << "beta: " << m_->included_coefficients() << endl;
      report_error(err.str());
    }

    uint n = g.nvars_possible();
    if(max_flips_ > 0) n = std::min<int>(n, max_flips_);
    for(uint i=0; i<n; ++i){
      logp = mcmc_one_flip(g, indx[i], logp);
    }
    m_->coef().set_inc(g);
  }

  double BLSSS::mcmc_one_flip(Selector &mod, uint which_var, double logp_old){
    mod.flip(which_var);
    double logp_new = log_model_prob(mod);
    double u = runif_mt(rng(), 0,1);
    if(log(u) > logp_new - logp_old){
      mod.flip(which_var);  // reject draw
      return logp_old;
    }
    return logp_new;
  }

}
