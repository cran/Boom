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
#include <Models/PosteriorSamplers/BetaBinomialSampler.hpp>
#include <distributions.hpp>

namespace BOOM{

  typedef BetaBinomialSampler BBS;

  BBS::BetaBinomialSampler(BinomialModel *mod,
               Ptr<BetaModel> pri, RNG &seeding_rng)
    : PosteriorSampler(seeding_rng),
      mod_(mod),
      pri_(pri)
  {}

  void BBS::draw(){
    double a = pri_->a();
    double b = pri_->b();
    double nyes = mod_->suf()->sum();
    double n = mod_->n() * mod_->suf()->nobs();
    double nno = n - nyes;
    double p;
    int ntries = 0;
    do {
      // In most cases this do loop will finish without repeating.  It
      // exists to guard against cases where rbeta returns values of p
      // on the boundary, or nan.
      p = rbeta_mt(rng(), a + nyes, b+nno);
      if (++ntries > 500) {
        const double epsilon = std::numeric_limits<double>::epsilon();
        if (p >= 1.0) {
          p = 1.0 - epsilon;
        } else if (p <= 0.0) {
          p = epsilon;
        } else if (!std::isfinite(p)) {
          ostringstream err;
          err << "Too many attempts in BetaBinomialSampler::draw()." << endl
              << "a = " << a << endl
              << "b = " << b << endl
              << "a + nyes = " << a + nyes << endl
              << "b + nno  = " << b + nno << endl
              << "last simulated value of p: " << p << endl;
          report_error(err.str());
        }
      }
    } while(p <= 0 || p >= 1 || !std::isfinite(p));
    mod_->set_prob(p);
  }

  double BBS::logpri()const{
    double p = mod_->prob();
    return pri_->logp(p);
  }

  void BBS::find_posterior_mode(double){
    double a = pri_->a();
    double b = pri_->b();
    double y = mod_->suf()->sum() + a;
    double n = mod_->suf()->nobs() + a + b;
    mod_->set_prob( (y - 1) / (n - 2) );
  }

}
