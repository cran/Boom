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

#include <Models/Glm/PosteriorSamplers/MLVS_data_imputer.hpp>
#include <boost/ref.hpp>

#include <cpputil/math_utils.hpp>
#include <cpputil/lse.hpp>

#include <stats/logit.hpp>

#include <distributions.hpp>
#include <cmath>

#ifndef NO_BOOST_THREADS
#include <boost/thread.hpp>

using boost::thread_group;
using boost::thread;
#endif

namespace BOOM{

  typedef MlvsDataImputer MDI;
  MDI::MlvsDataImputer(
      MultinomialLogitModel *Mod,
      Ptr<MultinomialLogitCompleteDataSufficientStatistics> Suf,
      uint nthreads){
#ifdef NO_BOOST_THREADS
      imp = new mlvs_impute::MDI_unthreaded(Mod, Suf);
#else
    if(nthreads<=1){
      imp = new mlvs_impute::MDI_unthreaded(Mod, Suf);
    }else{
      imp = new mlvs_impute::MDI_threaded(Mod, Suf, nthreads);
    }
#endif
  }

  void MDI::draw(){ imp->draw();}

  //______________________________________________________________________

  namespace mlvs_impute{
    typedef MDI_worker MDIW;

    MDI_unthreaded::MDI_unthreaded(
        MultinomialLogitModel *m,
        Ptr<MultinomialLogitCompleteDataSufficientStatistics> s)
        : mlm(m),
          suf(s),
          imp(mlm, s)
    {}
    void MDI_unthreaded::draw(){ imp(); }
    //======================================================================
#ifndef NO_BOOST_THREADS
    MDI_threaded::MDI_threaded(
        MultinomialLogitModel *m,
        Ptr<MultinomialLogitCompleteDataSufficientStatistics> s,
        uint nthreads)
        : mlm(m),
          suf(s)
    {
      for(uint i=0; i<nthreads; ++i){
        NEW(MDI_worker, worker)(m, s, i, nthreads);
        crew.push_back(worker);
      }
    }

    void MDI_threaded::draw(){
      boost::thread_group tg;
      for(uint i=0; i<crew.size(); ++i){
        tg.add_thread(new thread(boost::ref(*crew[i])));
      }
      tg.join_all();

      suf->clear();
      for(uint i=0; i<crew.size(); ++i){
        Ptr<MultinomialLogitCompleteDataSufficientStatistics> s = crew[i]->suf();
        suf->add(crew[i]->suf());
      }
    }
#endif
    //======================================================================

    unsigned long getseed(){
       double u = runif() * std::numeric_limits<unsigned long>::max();
       // convert from double to long long (using llround), and then
       // from long long to unsigned long.
       unsigned long ans(llround(u));
       return ans;
     }
    //======================================================================
    MDIW::MDI_worker(
        MultinomialLogitModel *mod,
        Ptr<MultinomialLogitCompleteDataSufficientStatistics> s,
        uint tid, uint nt)
        : mlm(mod),
          suf_(nt<=1 ? s : s->clone()),
          thread_id(tid),
          nthreads(nt),
          mu_(Vec("5.09 3.29 1.82 1.24 0.76 0.39 0.04 -0.31 -0.67  -1.06")),
          sigsq_inv_(pow(Vec(
              "4.5 2.02 1.1 0.42 0.2 0.11 0.08 0.08 0.09 0.15"),-1)),
          sd_(pow(sigsq_inv_,-0.5)),
          logpi_(log(Vec(
              "0.004 0.04 0.168 0.147 0.125 0.101 0.104 0.116 0.107 0.088"))),
          log_sampling_probs_(mlm->log_sampling_probs()),
          downsampling_ (log_sampling_probs_.size() == mlm->Nchoices()),
          post_prob_ (logpi_),
          u(mod->Nchoices()),
          eta(u),
          wgts(u),
          thisX(new Mat(1,1)),
          rng(getseed())
    {
      rng.seed();
    }
    //----------------------------------------------------------------------
    void MDIW::impute_u(Ptr<ChoiceData> dp){
      mlm->fill_eta(*dp, eta);      // eta+= downsampling_logprob
      if(downsampling_) eta += log_sampling_probs_;  //
      uint M = mlm->Nchoices();
      uint y = dp->value();
      assert(y<M);
      double loglam = lse(eta);
      double logzmin = rlexp_mt(rng, loglam);
      u[y] = - logzmin;
      for(uint m=0; m<M; ++m){
        if(m!=y){
          double tmp = rlexp_mt(rng, eta[m]);
          double logz = lse2(logzmin, tmp);
          u[m] = -logz;
        }
        uint k = unmix(u[m]-eta[m]);
        u[m] -= mu_[k];
        wgts[m] = sigsq_inv_[k];}}
    //----------------------------------------------------------------------
    uint MDIW::unmix(double u){
      uint K = post_prob_.size();
      for(uint k=0; k<K; ++k)
        post_prob_[k] = logpi_[k] + dnorm(u, mu_[k], sd_[k], true);
      post_prob_.normalize_logprob();
      return  rmulti_mt(rng, post_prob_);
    }
    //----------------------------------------------------------------------
    void MDIW::operator()(){
      const std::vector<Ptr<ChoiceData> > & dat(mlm->dat());
      suf_->clear();
      uint n = dat.size();
      uint i = thread_id;
      while(i < n){
        Ptr<ChoiceData> dp(dat[i]);
        impute_u(dp);
        suf_->update(dp, wgts, u);
        i+= nthreads;}
    }

    //----------------------------------------------------------------------

    void MDIW::seed(unsigned long s){ rng.seed(s); }

    const Ptr<MultinomialLogitCompleteDataSufficientStatistics>
    MDIW::suf() const {
      return suf_;
    }

  } // mdi_impute
} // namespace BOOM
