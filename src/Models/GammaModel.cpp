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
#include <Models/GammaModel.hpp>
#include <cmath>
#include <limits>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>
#include <distributions.hpp>
#include <cpputil/math_utils.hpp>
#include <Models/SufstatAbstractCombineImpl.hpp>

namespace BOOM{

  typedef GammaSuf GS;
  typedef GammaModelBase GMB;

  GS::GammaSuf()
      : sum_(0),
        sumlog_(0),
        n_(0)
  {}

  GS::GammaSuf(const GammaSuf &rhs)
    : Sufstat(rhs),
      SufstatDetails<DataType>(rhs),
      sum_(rhs.sum_),
      sumlog_(rhs.sumlog_),
      n_(rhs.n_)
  {}

  GS *GS::clone() const{return new GS(*this);}

  void GS::set(double sum, double sumlog, double n){
    // Check for impossible values.
    if (n > 0) {
      if (sum <= 0.0) {
        report_error("GammaSuf cannot have a negative sum if "
                     "it has a positive sample size");
      }
      // There is no minimum value that sumlog can achieve, because
      // any individual observation might be arbitrarily close to
      // zero, driving the sum of logs close to negative infinity.
      //
      // The sum of logs is maximized if each observation is the same
      // size.
      double ybar = sum / n;
      if (sumlog > n * log(ybar)) {
        report_error("GammaSuf was set with an impossibly large value "
                     "of sumlog.");
      }
    } else if (n < 0) {
      report_error("GammaSuf set to have a negative sample size.");
    } else {
      if (std::fabs(sum) > std::numeric_limits<double>::epsilon()
          || std::fabs(sumlog) > std::numeric_limits<double>::epsilon()) {
        report_error("All elements of GammaSuf must be zero if n == 0.");
      }
    }
    sum_ = sum;
    sumlog_ = sumlog;
    n_ = n;
  }

  void GS::clear(){sum_ = sumlog_ = n_=0;}

  void GS::Update(const DoubleData &dat){
    double x = dat.value();
    update_raw(x);
  }

  void GS::update_raw(double x){
    ++n_;
    sum_ += x;
    sumlog_ += log(x);
  }

  void GS::increment(double n, double sum, double sumlog) {
    n_ += n;
    sum_ += sum;
    sumlog_ += sumlog;
  }

  void GS::add_mixture_data(double y, double prob){
    n_ += prob;
    sum_ += prob * y;
    sumlog_ += prob * log(y);
  }

  double GS::sum()const{return sum_;}
  double GS::sumlog()const{return sumlog_;}
  double GS::n()const{return n_;}
  ostream & GS::display(ostream &out)const{
    out << "gamma::sum    = " << sum_ << endl
	<< "gamma::sumlog = " << sumlog_ <<  endl
	<< "gamma::n      = " << n_ << endl;
    return out;
  }


  void GS::combine(Ptr<GS> s){
    sum_ += s->sum_;
    sumlog_ += s->sumlog_;
    n_ += s->n_;
  }

  void GS::combine(const GS & s){
    sum_ += s.sum_;
    sumlog_ += s.sumlog_;
    n_ += s.n_;
  }

  GammaSuf * GS::abstract_combine(Sufstat *s){
    return abstract_combine_impl(this,s);    }

  Vector GS::vectorize(bool)const{
    Vector ans(3);
    ans[0] = sum_;
    ans[1] = sumlog_;
    ans[2] = n_;
    return ans;
  }

  Vector::const_iterator GS::unvectorize(Vector::const_iterator &v, bool){
    sum_ = *v;    ++v;
    sumlog_ = *v; ++v;
    n_ = *v;      ++v;
    return v;
  }

  Vector::const_iterator GS::unvectorize(const Vector &v, bool minimal){
    Vector::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  ostream & GS::print(ostream &out)const{
    return out << n_ << " " << sum_ << " " << sumlog_;
  }
  //======================================================================
  GMB::GammaModelBase()
    : DataPolicy(new GammaSuf())
  {}

  GMB::GammaModelBase(const GMB &rhs)
    : Model(rhs),
      DataPolicy(rhs),
      DiffDoubleModel(rhs),
      NumOptModel(rhs),
      EmMixtureComponent(rhs)
  {}

  double GMB::pdf(Ptr<Data> dp, bool logscale)const{
    double ans = logp(DAT(dp)->value());
    return logscale ? ans : exp(ans);}

  double GMB::pdf(const Data * dp, bool logscale)const{
    double ans = logp(DAT(dp)->value());
    return logscale ? ans : exp(ans);}

  double GMB::Logp(double x, double &g, double &h, uint nd) const{
     double a = alpha();
     double b = beta();
     double ans = dgamma(x, a,b,true);
     if(nd>0) g = (a-1)/x-b;
     if(nd>1) h = -(a-1)/(x*x);
     return ans;
  }
  double GMB::sim() const{
    return rgamma(alpha(), beta());}

  void GMB::add_mixture_data(Ptr<Data> dp, double prob){
    double y = DAT(dp)->value();
    suf()->add_mixture_data(y, prob);
  }

  //======================================================================

  GammaModel::GammaModel(double a, double b)
    : GMB(),
      ParamPolicy(new UnivParams(a), new UnivParams(b)),
      PriorPolicy()
  {
    if (a <= 0 || b <= 0) {
      report_error("Both parameters must be positive in the "
                   "GammaModel constructor.");
    }
  }

  GammaModel::GammaModel(double shape, double mean, int)
      : GMB(),
        ParamPolicy(new UnivParams(shape), new UnivParams(shape / mean)),
        PriorPolicy()
  {
    if (shape <= 0 || mean <= 0) {
      report_error("Both parameters must be positive in the "
                   "GammaModel constructor.");
    }
  }

  GammaModel::GammaModel(const GammaModel &rhs)
    : Model(rhs),
      GMB(rhs),
      ParamPolicy(rhs),
      PriorPolicy(rhs)
  {}

  GammaModel * GammaModel::clone()const{
    return new GammaModel(*this);}


  Ptr<UnivParams> GammaModel::Alpha_prm(){
    return ParamPolicy::prm1();
  }

  Ptr<UnivParams> GammaModel::Beta_prm(){
    return ParamPolicy::prm2();
  }

  const Ptr<UnivParams> GammaModel::Alpha_prm()const{
    return ParamPolicy::prm1();
  }

  const Ptr<UnivParams> GammaModel::Beta_prm()const{
    return ParamPolicy::prm2();
  }

  double GammaModel::alpha()const{
    return ParamPolicy::prm1_ref().value();
  }

  double GammaModel::beta()const{
    return ParamPolicy::prm2_ref().value();
  }

  void GammaModel::set_alpha(double a){
    if (a <= 0) {
      ostringstream err;
      err << "The 'a' parameter must be positive in GammaModel::set_alpha()."
          << endl
          << "Called with a = " << a << endl;
      report_error(err.str());
    }
    ParamPolicy::prm1_ref().set(a);
  }

  void GammaModel::set_beta(double b){
    if (b <= 0) {
      ostringstream err;
      err << "The 'b' parameter must be positive in GammaModel::set_beta()."
          << endl
          << "Called with b = " << b << endl;
      report_error(err.str());
    }
    ParamPolicy::prm2_ref().set(b);
  }

  void GammaModel::set_params(double a, double b){
    set_alpha(a);
    set_beta(b);}

  double GammaModel::mean()const{
    return alpha() / beta();
  }

  inline double bad_gamma_loglike(double a,double b, Vector &g, Matrix &h, uint nd){
    if(nd>0){
      g[0] = (a <=0) ? -(a+1) : 0;
      g[1] = (b <= 0) ? -(b+1) : 0;
      if(nd>1) h.set_diag(-1);
    }
    return BOOM::negative_infinity();
  }

  inline double bad_gamma_loglike(double a, double b, Vector *g, Matrix *h){
    if (g) {
      (*g)[0] = (a <=0) ? -(a+1) : 0;
      (*g)[1] = (b <= 0) ? -(b+1) : 0;
    }
    if (h) {
      h->set_diag(-1);
    }
    return negative_infinity();
  }

  double GammaModel::Loglike(const Vector &ab, Vector &g, Matrix &h, uint nd) const{
    if (ab.size() != 2) {
      report_error("Wrong size argument.");
    }
    double n = suf()->n();
    double sum =suf()->sum();
    double sumlog = suf()->sumlog();
    double a = ab[0];
    double b = ab[1];
    if(a<=0 || b<=0) return bad_gamma_loglike(a, b,g,h,nd);

    double logb = log(b);
    double ans = n*(a*logb -lgamma(a))  + (a-1)*sumlog - b*sum;

    if(nd>0){
      g[0] = n*( logb -digamma(a) ) + sumlog;
      g[1] = n*a/b -sum;
      if(nd>1){
 	h(0,0) = -n*trigamma(a);
 	h(1,0) = h(0,1) = n/b;
 	h(1,1) = -n*a/(b*b);}}
    return ans;
  }

  double GammaModel::loglikelihood(double a, double b)const{
    if(a<=0 || b<=0) return negative_infinity();
    double n = suf()->n();
    double sum =suf()->sum();
    double sumlog = suf()->sumlog();
    double logb = log(b);
    return n*(a*logb -lgamma(a))  + (a-1)*sumlog - b*sum;
  }

  double GammaModel::loglikelihood_full(const Vector &ab, Vector *g, Matrix *h)const{
    if (length(ab) != 2) {
      report_error("GammaModel::loglikelihood expects an argument of length 2");
    }
    double a = ab[0];
    double b = ab[1];
    if(a<=0 || b<=0) return bad_gamma_loglike(a, b, g, h);

    double n = suf()->n();
    double sum =suf()->sum();
    double sumlog = suf()->sumlog();
    double logb = log(b);
    double ans = n*(a*logb -lgamma(a))  + (a-1)*sumlog - b*sum;

    if (g) {
      if (length(*g) != 2) {
        report_error("GammaModel::loglikelihood expects a gradient vector "
                     "of length 2");
      }
      (*g)[0] = n*( logb -digamma(a) ) + sumlog;
      (*g)[1] = n*a/b -sum;

      if (h) {
        if (nrow(*h) != 2 || ncol(*h) != 2) {
          report_error("GammaModel::loglikelihood expects a 2 x 2 "
                       "Hessian matrix");
        }
      }
    }
    return ans;
  }

  void GammaModel::mle(){
    // can get good starting values;
    double n = suf()->n();
    double sum= suf()->sum();
    double sumlog = suf()->sumlog();

    double ybar = sum/n;        // arithmetic mean
    double gm = exp(sumlog/n);  // geometric mean
    double ss=0;
    for(uint i=0; i<dat().size(); ++i)
      ss+= pow(dat()[i]->value()-ybar, 2);
    if ( (ss > 0) && (n > 1) ) {
      double v = ss/(n-1);

      // method of moments estimates
      double b = ybar/v;
      double a = ybar*b;

      // one step newton refinement:
      // a = ybar *b;
      // b - exp(psi(ybar*b))/gm = 0
      double tmp = exp(digamma(ybar*b))/gm;
      double f =  b - tmp;
      double g = 1 - tmp*trigamma(ybar*b) * ybar;

      b-= f/g;
      a = b*ybar;
      set_params(a,b);
    }
    NumOptModel::mle();
  }

  //======================================================================

}
