/*
  Copyright (C) 2006 Steven L. Scott

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
#include <Models/PosteriorSamplers/MvnVarSampler.hpp>
#include <Models/MvnModel.hpp>
#include <Models/WishartModel.hpp>
#include <distributions.hpp>
namespace BOOM{

  MvnVarSampler::MvnVarSampler(MvnModel *m, RNG &seeding_rng)
    : PosteriorSampler(seeding_rng),
      mvn_(m),
      pdf_(new UnivParams(0.0))
  {
    SpdMatrix sumsq = m->Sigma().Id();
    sumsq.set_diag(0.0);
    pss_ = new SpdParams(sumsq);
  }

  MvnVarSampler::MvnVarSampler(MvnModel *m, double df, const SpdMatrix &sumsq,
                               RNG &seeding_rng)
    : PosteriorSampler(seeding_rng),
      mvn_(m),
      pdf_(new UnivParams(df)),
      pss_(new SpdParams(sumsq))
  {}

  MvnVarSampler::MvnVarSampler(MvnModel *m, const WishartModel &siginv_prior,
                               RNG &seeding_rng)
    : PosteriorSampler(seeding_rng),
      mvn_(m),
      pdf_(siginv_prior.Nu_prm()),
      pss_(siginv_prior.Sumsq_prm())
  {}

  double MvnVarSampler::logpri()const{
    const SpdMatrix & siginv(mvn_->siginv());
    return dWish(siginv, pss_->var(), pdf_->value(), true);
  }

  void MvnVarSampler::draw(){
    Ptr<MvnSuf> s = mvn_->suf();
    double df = pdf_->value() + s->n();
    SpdMatrix S = s->center_sumsq(mvn_->mu());
    S += pss_->value();
    S = rWish(df, S.inv());
    Ptr<SpdParams> sp = mvn_->Sigma_prm();
    sp->set_ivar(S);
  }

  //======================================================================

  MvnConjVarSampler::MvnConjVarSampler(MvnModel *m, RNG &seeding_rng)
      : MvnVarSampler(m, seeding_rng) {}

  MvnConjVarSampler::MvnConjVarSampler(MvnModel *m, double df, const SpdMatrix &sumsq,
                                       RNG &seeding_rng)
      : MvnVarSampler(m, df, sumsq, seeding_rng) {}

  MvnConjVarSampler::MvnConjVarSampler(MvnModel *m,
                                       const WishartModel &siginv_prior,
                                       RNG &seeding_rng)
      : MvnVarSampler(m, siginv_prior, seeding_rng) {}

  void MvnConjVarSampler::draw(){
    Ptr<MvnSuf> s = mvn()->suf();
    double df = pdf()->value() + s->n() - 1;
    SpdMatrix S = s->center_sumsq(s->ybar());
    S += pss()->value();
    S = rWish(df, S.inv());
    Ptr<SpdParams> sp = mvn()->Sigma_prm();
    sp->set_ivar(S);
  }

} // namespace BOOM
