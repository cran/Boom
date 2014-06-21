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

#ifndef WISHART_MODEL_H
#define WISHART_MODEL_H

#include <Models/ModelTypes.hpp>
#include <Models/SpdParams.hpp>
#include <Models/SpdModel.hpp>
#include <Models/Sufstat.hpp>
#include <Models/Policies/SufstatDataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/Policies/ParamPolicy_2.hpp>

namespace BOOM{
  class WishartSuf : public SufstatDetails<SpdData> {
  public:
    WishartSuf(uint dim);
    WishartSuf(const WishartSuf &sf);
    WishartSuf *clone() const;

    void clear();
    void Update(const SpdData &d);
    double n()const{return n_;}
    double sumldw()const{return sumldw_;}
    const Spd & sumW()const{return sumW_;}
    void combine(Ptr<WishartSuf>);
    void combine(const WishartSuf &);
    WishartSuf * abstract_combine(Sufstat *s);
    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true);
    virtual ostream &print(ostream &out)const;
  private:
    double n_;
    double sumldw_;
    Spd sumW_;
  };
  //======================================================================
  class WishartModel :
    public ParamPolicy_2<UnivParams, SpdParams>,
    public SufstatDataPolicy<SpdData, WishartSuf>,
    public PriorPolicy,
    public dLoglikeModel,
    public SpdModel
  {
  public:
    WishartModel(uint dim);
    WishartModel(uint dim, double prior_df, double diagonal_variance);
    WishartModel(double prior_df, const Spd &prior_var_est);
    WishartModel(const WishartModel &m);

    WishartModel *clone() const;

    virtual void initialize_params();

    Ptr<UnivParams> Nu_prm();
    Ptr<SpdParams> Sumsq_prm();
    const Ptr<UnivParams> Nu_prm()const;
    const Ptr<SpdParams> Sumsq_prm()const;

    const double & nu() const;
    const Spd &sumsq() const;
    void set_nu(double);
    void set_sumsq(const Spd &);

    Spd simdat();
    int dim()const {return sumsq().nrow();}
    void mle0();
    void mle1();

    virtual double logp(const Spd &W) const;
    double loglike() const;
    double dloglike(Vec &g)const;
    double Loglike(Vec &g, uint  nd)const;
  };
  //======================================================================


}
#endif // WISHART_MODEL_H
