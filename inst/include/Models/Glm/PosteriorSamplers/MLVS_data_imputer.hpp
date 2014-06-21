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

#ifndef BOOM_MLVS_DATA_IMPUTER_HPP
#define BOOM_MLVS_DATA_IMPUTER_HPP

#include <Models/Glm/ChoiceData.hpp>
#include <Models/Glm/MultinomialLogitModel.hpp>
#include <Models/Glm/PosteriorSamplers/MultinomialLogitCompleteDataSuf.hpp>
#include <distributions/rng.hpp>

namespace BOOM{

  namespace mlvs_impute{
    class MDI_base;
    class MDI_worker;
  }

  class MlvsDataImputer : private RefCounted{
  public:
    MlvsDataImputer(MultinomialLogitModel *Mod,
                    Ptr<MultinomialLogitCompleteDataSufficientStatistics> Suf,
                    uint nthreads);
    void draw();

    friend void intrusive_ptr_add_ref(MlvsDataImputer *d){
      d->up_count();}
    friend void intrusive_ptr_release(MlvsDataImputer *d){
      d->down_count(); if(d->ref_count()==0) delete d;}

  private:
    Ptr<mlvs_impute::MDI_base> imp;
  };

  //______________________________________________________________________

  namespace mlvs_impute{

    class MDI_worker : private RefCounted {
    public:

      friend void intrusive_ptr_add_ref(MDI_worker *d){d->up_count();}
      friend void intrusive_ptr_release(MDI_worker *d){
        d->down_count(); if(d->ref_count()==0) delete d;}

      MDI_worker(MultinomialLogitModel *mod,
                 Ptr<MultinomialLogitCompleteDataSufficientStatistics> s,
                 uint Thread_id=0,
                 uint Nthreads=1);
      void impute_u(Ptr<ChoiceData> dp);
      uint unmix(double u);
      const Ptr<MultinomialLogitCompleteDataSufficientStatistics> suf()const;
      void operator()();
      void seed(unsigned long);

    private:
      MultinomialLogitModel *mlm;
      Ptr<MultinomialLogitCompleteDataSufficientStatistics> suf_;

      const uint thread_id;
      const uint nthreads;
      const Vec mu_;        // mean for EV approx
      const Vec sigsq_inv_; // inverse variance for EV approx
      const Vec sd_;        // standard deviations for EV approx
      const Vec logpi_;     // log of mixing weights for EV approx
      const Vec & log_sampling_probs_;
      const bool downsampling_;

      Vec post_prob_;
      Vec u;
      Vec eta;
      Vec wgts;

      boost::shared_ptr<Mat> thisX;
      RNG rng;
    };
    //======================================================================
    class MDI_base : private RefCounted{
    public:
      friend void intrusive_ptr_add_ref(MDI_base *d){d->up_count();}
      friend void intrusive_ptr_release(MDI_base *d){
        d->down_count(); if(d->ref_count()==0) delete d;}

      virtual void draw()=0;
      virtual ~MDI_base(){}
    };

    //======================================================================
    class MDI_unthreaded : public MDI_base {
    public:
      MDI_unthreaded(MultinomialLogitModel *m,
                     Ptr<MultinomialLogitCompleteDataSufficientStatistics> s);
      virtual void draw();
    private:
      MultinomialLogitModel *mlm;
      Ptr<MultinomialLogitCompleteDataSufficientStatistics> suf;
      MDI_worker imp;
    };

    //======================================================================
    class MDI_threaded : public MDI_base {
    public:
      MDI_threaded(MultinomialLogitModel *m,
                   Ptr<MultinomialLogitCompleteDataSufficientStatistics> s,
                   uint nthreads);
      virtual void draw();
    private:
      MultinomialLogitModel *mlm;
      Ptr<MultinomialLogitCompleteDataSufficientStatistics> suf;
      std::vector<Ptr<MDI_worker> > crew;
    };

  }
}

#endif// BOOM_MLVS_DATA_IMPUTER_HPP
