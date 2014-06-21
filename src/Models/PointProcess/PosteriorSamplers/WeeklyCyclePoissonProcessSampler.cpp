/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#include <Models/PointProcess/PosteriorSamplers/WeeklyCyclePoissonProcessSampler.hpp>
#include <distributions.hpp>

namespace BOOM{

  namespace{
    typedef WeeklyCyclePoissonProcessSampler SAM;
  }

  SAM::WeeklyCyclePoissonProcessSampler(
      WeeklyCyclePoissonProcess *model,
      Ptr<GammaModelBase> average_daily_rate_prior,
      Ptr<DirichletModel> day_of_week_prior,
      Ptr<DirichletModel> weekday_hourly_prior,
      Ptr<DirichletModel> weekend_hourly_prior)
      : model_(model),
        average_daily_rate_prior_(average_daily_rate_prior),
        day_of_week_prior_(day_of_week_prior),
        weekday_hourly_prior_(weekday_hourly_prior),
        weekend_hourly_prior_(weekend_hourly_prior)
  {}

  void SAM::draw(){
    draw_average_daily_rate();
    draw_daily_pattern();
    draw_weekday_hourly_pattern();
    draw_weekend_hourly_pattern();
  }

  double SAM::logpri()const{
    double ans = average_daily_rate_prior_->logp(model_->average_daily_rate());
    ans += day_of_week_prior_->logp(model_->day_of_week_pattern());
    ans += weekday_hourly_prior_->logp(model_->weekday_hourly_pattern());
    ans += weekend_hourly_prior_->logp(model_->weekend_hourly_pattern());
    return ans;
  }

  void SAM::draw_average_daily_rate(){
    double a = sum(model_->suf()->count()) +
        average_daily_rate_prior_->alpha();
    double b = average_daily_rate_prior_->beta();
    const Vec &daily(model_->day_of_week_pattern());
    const Vec &weekend(model_->weekend_hourly_pattern());
    const Vec &weekday(model_->weekday_hourly_pattern());
    const Mat &exposure(model_->suf()->exposure());
    for(int d = 0; d < 6; ++d){
      const Vec &hourly((d==Sat || d==Sun) ? weekend : weekday);
      for(int hour = 0; hour < 24; ++hour){
        b += daily[d] * hourly[hour] * exposure(d, hour);
      }
    }
    double lambda = rgamma_mt(rng(), a, b);
    model_->set_average_daily_rate(lambda);
  }

  void SAM::draw_daily_pattern(){
    Vec nu = model_->suf()->daily_event_count() + day_of_week_prior_->nu();
    Vec cand = rdirichlet_mt(rng(), nu);
    Vec orig = model_->day_of_week_pattern() / 7;
    double denom = model_->loglike() - ddirichlet(orig, nu, true);
    model_->set_day_of_week_pattern(cand * 7);
    double num = model_->loglike() - ddirichlet(cand, nu, true);
    ++daily_pattern_attempts_;
    double logu = log(runif_mt(rng()));
    if(logu > num - denom){
      // MH step failed
      model_->set_day_of_week_pattern(orig * 7);
    }else{
      ++daily_pattern_successes_;
    }
  }

  void SAM::draw_weekend_hourly_pattern(){
    Vec nu = model_->suf() -> weekend_hourly_event_count() +
        weekend_hourly_prior_->nu();
    Vec cand = rdirichlet_mt(rng(), nu);
    Vec orig = model_->weekend_hourly_pattern() / 24;
    double denom = model_->loglike() - ddirichlet(orig, nu, true);
    model_->set_weekend_hourly_pattern(cand * 24);
    double num = model_->loglike() - ddirichlet(cand, nu, true);
    ++weekend_hourly_attempts_;
    double logu = log(runif_mt(rng()));
    if(logu > num - denom){
      model_->set_weekend_hourly_pattern(orig * 24);
    }else{
      ++weekend_hourly_successes_;
    }
  }

  void SAM::draw_weekday_hourly_pattern(){
    Vec nu = model_->suf() -> weekday_hourly_event_count() +
        weekday_hourly_prior_->nu();
    Vec cand = rdirichlet_mt(rng(), nu);
    Vec orig = model_->weekday_hourly_pattern() / 24;
    double denom = model_->loglike() - ddirichlet(orig, nu, true);
    model_->set_weekday_hourly_pattern(cand * 24);
    double num = model_->loglike() - ddirichlet(cand, nu, true);
    ++weekday_hourly_attempts_;
    double logu = log(runif_mt(rng()));
    if(logu > num - denom){
      model_->set_weekday_hourly_pattern(orig * 24);
    }else{
      ++weekday_hourly_successes_;
    }
  }


}
