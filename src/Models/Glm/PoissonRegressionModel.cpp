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

#include <Models/Glm/PoissonRegressionModel.hpp>
#include <distributions.hpp>

namespace BOOM {

  PoissonRegressionModel::PoissonRegressionModel(int xdim)
      : ParamPolicy(new GlmCoefs(xdim))
  {}

  PoissonRegressionModel::PoissonRegressionModel(const Vec &beta)
      : ParamPolicy(new GlmCoefs(beta))
  {}

  PoissonRegressionModel * PoissonRegressionModel::clone()const{
    return new PoissonRegressionModel(*this);}

  GlmCoefs & PoissonRegressionModel::coef(){
    return ParamPolicy::prm_ref();
  }
  const GlmCoefs & PoissonRegressionModel::coef()const{
    return ParamPolicy::prm_ref();
  }

  Ptr<GlmCoefs> PoissonRegressionModel::coef_prm(){
    return ParamPolicy::prm();}

  const Ptr<GlmCoefs> PoissonRegressionModel::coef_prm()const{
    return ParamPolicy::prm();}


  double PoissonRegressionModel::log_likelihood(
      const Vec &beta, Vec *g, Mat *h)const{
    // L = (E *lambda)^y exp(-E*lambda)
    //   ell = y * (log(E) + log(lambda)) - E*exp(x * beta)
    //       = yXbeta - E*exp(Xbeta)
    // dell  = (y - E*lambda) * x
    // ddell = -lambda * x * x'
    double ans = 0;
    const std::vector<Ptr<PoissonRegressionData> > &data(dat());
    if (g) {
      g->resize(xdim());
      (*g) = 0.0;
      if (h) {
        h->resize(xdim(), xdim());
        (*h) = 0.0;
      }
    }

    for(int i = 0; i < data.size(); ++i){
      const Vec &x(data[i]->x());
      double eta = beta.dot(x);
      int y = data[i]->y();
      double lambda = exp(eta);
      double exposure = data[i]->exposure();
      ans += dpois(y, exposure * lambda, true);
      if(g){
        g->axpy(x, (y - exposure * lambda));
        if(h){
          h->add_outer(x, x, -lambda);
        }
      }
    }
    return ans;
  }

  double PoissonRegressionModel::Loglike(Vec &g, Mat &h, uint nd)const{
    Vec *gp = NULL;
    Mat *hp = NULL;
    if (nd > 0) gp = &g;
    if (nd > 1) hp = &h;
    return log_likelihood(Beta(), gp, hp);
  }

  double PoissonRegressionModel::pdf(const Data *dp, bool logscale)const{
    // const PoissonRegressionData *d(
    //     dynamic_cast<const PoissonRegressionData *>(dp));
    const PoissonRegressionData *d(DAT(dp));
    double ans = logp(*d);
    return logscale ? ans : exp(ans);
  }

  double PoissonRegressionModel::logp(const PoissonRegressionData &data)const{
    double lambda = exp(predict(data.x()));
    return dpois(data.y(), data.exposure() * lambda, true);
  }

} // namespace BOOM
