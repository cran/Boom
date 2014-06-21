/*
  Copyright (C) 2005 Steven L. Scott

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

#include <Models/PoissonModel.hpp>
#include <cmath>
#include <distributions.hpp>
#include <Models/GammaModel.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <Models/PosteriorSamplers/PoissonGammaSampler.hpp>
#include <Models/SufstatAbstractCombineImpl.hpp>

namespace BOOM{

  PoissonSuf::PoissonSuf() : sum_(0.0), n_(0.0), lognc_(0.0){}
  PoissonSuf *PoissonSuf::clone() const{return new PoissonSuf(*this);}
  void PoissonSuf::clear(){sum_=n_=lognc_=0;}

  PoissonSuf::PoissonSuf(double event_count, double exposure)
      : sum_(event_count), n_(exposure), lognc_(0) {}

  PoissonSuf::PoissonSuf(const PoissonSuf &rhs)
    : Sufstat(rhs),
      SufstatDetails<DataType>(rhs)
  {
    sum_ = rhs.sum_;
    n_ = rhs.n_;
    lognc_ = rhs.lognc_;
  }

  void PoissonSuf::set(double event_count, double exposure){
    sum_ = event_count;
    n_ = exposure;
    lognc_ = 0;
  }

  void PoissonSuf::Update(const DataType  &X){
    int x = X.value();
    sum_+=x;
    lognc_+= lgamma(x+1);
    n_+=1.0;
  }

  void PoissonSuf::add_mixture_data(double y, double prob){
    n_ += prob;
    lognc_ += log(prob) + lgamma(y+1);
    sum_ += (prob*y);
  }

  double PoissonSuf::sum()const{return sum_;}
  double PoissonSuf::n()const{return n_;}
  double PoissonSuf::lognc()const{return lognc_;}

  void PoissonSuf::combine(Ptr<PoissonSuf> s){
    sum_ += s->sum_;
    n_ += s->n_;
    lognc_ += s->lognc_;
  }
  void PoissonSuf::combine(const PoissonSuf & s){
    sum_ += s.sum_;
    n_ += s.n_;
    lognc_ += s.lognc_;
  }

  PoissonSuf * PoissonSuf::abstract_combine(Sufstat *s){
    return abstract_combine_impl(this,s); }

  Vec PoissonSuf::vectorize(bool)const{
    Vec ans(3);
    ans[0] = sum_;
    ans[1] = n_;
    ans[2] = lognc_;
    return ans;
  }

  Vec::const_iterator PoissonSuf::unvectorize(Vec::const_iterator &v, bool){
    sum_ = *v;   ++v;
    n_ = *v;     ++v;
    lognc_ = *v; ++v;
    return v;
  }

  Vec::const_iterator PoissonSuf::unvectorize(const Vec &v,
                                              bool minimal){
    Vec::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  ostream &PoissonSuf::print(ostream &out)const{
    return out << sum_ << " " << n_;
  }

  //======================================================================

  PoissonModel::PoissonModel(double lam)
    : ParamPolicy(new UnivParams(lam)),
      DataPolicy(new PoissonSuf()),
      PriorPolicy()
  {}

  PoissonModel::PoissonModel(const std::vector<uint> &raw)
    : ParamPolicy(new UnivParams(1.0)),
      DataPolicy(new PoissonSuf()),
      PriorPolicy()
  {
    uint n = raw.size();
    for(uint i=0; i<n; ++i){
      NEW(DataType, dp)(raw[i]);
      this->add_data(dp);
    }
    mle();
  }

  PoissonModel::PoissonModel(const PoissonModel &rhs)
    : Model(rhs),
      MLE_Model(rhs),
      ParamPolicy(rhs),
      DataPolicy(rhs),
      PriorPolicy(rhs),
      NumOptModel(rhs),
      MixtureComponent(rhs)
  {}

  PoissonModel * PoissonModel::clone()const{
    return new PoissonModel(*this);}

  void PoissonModel::mle(){
    double n = suf()->n();
    double s=suf()->sum();
    if(n>0) set_lam(s/n);
    else set_lam(1.0);
  }

  double PoissonModel::Loglike(Vec &g, Mat &h, uint nd)const{
    double lam = this->lam();
    Ptr<PoissonSuf> s = suf();
    double sm = s->sum();
    double n = s->n();
    double ans = sm*log(lam) - n*lam - s->lognc();
    if(nd>0){
      g[0] = sm/lam-n;
      if(nd>1) h(0,0) = -sm/(lam*lam);
    }
    return ans;
  }

  Ptr<UnivParams> PoissonModel::Lam(){
    return ParamPolicy::prm();}
  const Ptr<UnivParams> PoissonModel::Lam()const{
    return ParamPolicy::prm();}

  double PoissonModel::lam()const{return Lam()->value();}
  void PoissonModel::set_lam(double x){Lam()->set(x);}

  double PoissonModel::pdf(uint x, bool logscale) const{
    return dpois(x, lam(), logscale); }

  double PoissonModel::pdf(Ptr<Data> dp, bool logscale) const{
    return dpois(DAT(dp)->value(), lam(), logscale); }

  double PoissonModel::pdf(const Data * dp, bool logscale) const{
    return dpois(DAT(dp)->value(), lam(), logscale); }

  double PoissonModel::mean()const{return lam();}
  double PoissonModel::var()const{return lam();}
  double PoissonModel::sd()const{return sqrt(lam());}
  double PoissonModel::simdat()const{
    return rpois(lam());}

  void PoissonModel::add_mixture_data(Ptr<Data> dp, double prob){
    double y = DAT(dp)->value();
    suf()->add_mixture_data(y, prob);
  }

}
