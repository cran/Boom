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

#include <Models/BinomialModel.hpp>
#include <cassert>
#include <distributions.hpp>
#include <cpputil/math_utils.hpp>
#include <Models/SufstatAbstractCombineImpl.hpp>
#include <Models/PosteriorSamplers/BetaBinomialSampler.hpp>

namespace BOOM{

  typedef BinomialSuf BS;
  typedef BinomialModel BM;

  BS::BinomialSuf()
    : SufTraits(),
      sum_(0),
      nobs_(0)
  {}

  BS::BinomialSuf(const BS &rhs)
    : Sufstat(rhs),
      SufTraits(rhs),
      sum_(rhs.sum_),
      nobs_(rhs.nobs_)
  {}

  BS * BS::clone()const{return new BS(*this);}

  void BS::set(double sum, double observation_count){
    nobs_ = observation_count;
    sum_ = sum;
  }

  void BS::clear(){ nobs_ = sum_ = 0;}

  void BS::Update(const IntData &d){
    int y = d.value();
    sum_ += y;
    nobs_ += 1;
  }

  void BS::update_raw(double y){
    sum_ += y;
    nobs_ += 1;
  }

  void BS::batch_update(double n, double y){
    sum_ += y;
    nobs_ += n;
  }

  void BS::add_mixture_data(double y, double prob){
    sum_ += y*prob;
    nobs_ += prob;
  }

  double BS::sum()const{return sum_;}
  double BS::nobs()const{return nobs_;}

  void BS::combine(Ptr<BS> s){
    sum_ += s->sum_;
    nobs_ += s->nobs_;
  }
  void BS::combine(const BS & s){
    sum_ += s.sum_;
    nobs_ += s.nobs_;
  }

  BinomialSuf * BS::abstract_combine(Sufstat *s){
    return abstract_combine_impl(this, s);}


  Vector BS::vectorize(bool)const{
    Vector ans(2);
    ans[0] = sum_;
    ans[1] = nobs_;
    return ans;
  }

  Vector::const_iterator BS::unvectorize(Vector::const_iterator &v,
                                      bool){
    sum_ = *v;  ++v;
    nobs_ = *v; ++v;
    return v;
  }

  Vector::const_iterator BS::unvectorize(const Vector &v, bool minimal){
    Vector::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  ostream & BS::print(ostream &out)const{
    return out << sum_ << " " << nobs_;
  }

  BM::BinomialModel(uint n, double p)
    : ParamPolicy(new UnivParams(p)),
      DataPolicy(new BS),
      NumOptModel(),
      n_(n)
  {
    assert(n>0);
  }

  BM::BinomialModel(const BM & rhs)
    : Model(rhs),
      ParamPolicy(rhs),
      DataPolicy(rhs),
      PriorPolicy(rhs),
      NumOptModel(rhs),
      n_(rhs.n_)
  {}

  BM * BM::clone()const{return new BM(*this);}

  void BM::mle(){
    double p = suf()->sum()/(n_*suf()->nobs());
    set_prob(p);
  }

  uint BM::n()const{return n_;}
  double BM::prob()const{ return Prob_prm()->value();}
  void BM::set_prob(double p){
    if (p < 0 || p > 1) {
      std::ostringstream err;
      err << "The argument to BinomialModel::set_prob was " << p
          << ", but a probability must be in the range [0, 1]."
          << endl;
      report_error(err.str());
    }
    Prob_prm()->set(p);
  }

  Ptr<UnivParams> BM::Prob_prm(){ return ParamPolicy::prm();}
  const Ptr<UnivParams> BM::Prob_prm()const{ return ParamPolicy::prm();}

  double BM::Loglike(const Vector &probvec, Vector &g, Matrix &h, uint nd)const{
    if (probvec.size() != 1) {
      report_error("Wrong size argument.");
    }
    double p = probvec[0];
    if (p < 0 || p > 1) {
      return negative_infinity();
    }
    double logp = log(p);
    double logp2 = log(1-p);

    double ntrials = n_ * suf()->nobs();
    double success = n_*suf()->sum();
    double fail = ntrials - success;

    double ans =  success * logp + fail * logp2;

    if(nd>0){
      double q = 1-p;
      g[0] = (success - p*ntrials)/(p*q);
      if(nd>1){
        h(0,0) = -1*(success/(p*p)  + fail/(q*q));
      }
    }
    return ans;
  }

  double BM::pdf(uint x,  bool logscale)const{
    if(x>n_)
      return logscale ? BOOM::negative_infinity() : 0;
    if(n_==1){
      double p = x==1 ? prob() : 1-prob();
      return logscale ? log(p) : p;
    }
    return dbinom(x,n_, prob(), logscale);
  }

  double BM::pdf(Ptr<Data> dp, bool logscale)const{
    return pdf(DAT(dp)->value(), logscale);}

  double BM::pdf(const Data * dp, bool logscale)const{
    return pdf(DAT(dp)->value(), logscale);}

  uint BM::sim()const{ return rbinom(n_, prob()); }

  void BM::add_mixture_data(Ptr<Data> dp, double prob){
    suf()->add_mixture_data(DAT(dp)->value(), prob);
  }

}
