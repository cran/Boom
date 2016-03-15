/*
  Copyright (C) 2008 Steven L. Scott

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

#ifndef BOOM_GAUSSIAN_MODEL_BASE_HPP
#define BOOM_GAUSSIAN_MODEL_BASE_HPP

#include <Models/ModelTypes.hpp>
#include <Models/DoubleModel.hpp>
#include <Models/Sufstat.hpp>
#include <Models/EmMixtureComponent.hpp>
#include <Models/Policies/SufstatDataPolicy.hpp>
#include <Models/DataTypes.hpp>

namespace BOOM{

  class GaussianSuf
    : public SufstatDetails<DoubleData>{
    double sum_, sumsq_, n_;
  public:
    // constructor
    GaussianSuf();
    GaussianSuf(double Sum, double Ssq, double N);
    GaussianSuf(const GaussianSuf &);
    GaussianSuf *clone() const override;

    void clear() override;
    void Update(const DoubleData &X) override;
    void update_raw(double y);

    void update_expected_value(
        double expected_sample_size,
        double expected_sum,
        double expected_sum_of_squares);

    // Remove the effect of observation y from the sufficient
    // statistics, as if it were dropped from the data set.
    void remove(double y);
    void add_mixture_data(double y, double prob);
    double sum()const;
    // sumsq returns the uncentered (raw) sum of squared y's: sum(y^2)
    double sumsq()const;
    // centered_sumsq returns sum((y - mu)^2).
    double centered_sumsq(double mu) const;
    double n()const;

    double ybar()const;
    double sample_var()const;

    GaussianSuf * abstract_combine(Sufstat *s) override;
    void combine(Ptr<GaussianSuf>);
    void combine(const GaussianSuf &);
    Vector vectorize(bool minimal=true)const override;
    Vector::const_iterator unvectorize(Vector::const_iterator &v,
                                            bool minimal=true) override;
    Vector::const_iterator unvectorize(const Vector &v,
                                            bool minimal=true) override;
    ostream &print(ostream &out)const override;
  };
  //======================================================================
  class GaussianModelBase
      : public SufstatDataPolicy<DoubleData, GaussianSuf>,
        public DiffDoubleModel,    // promises  Logp(x,g,h,nd);
        public NumOptModel,        // promises Loglike(g,h,nd), and mle();
        public EmMixtureComponent  // promises add_mixture_data
  {
   public:
    GaussianModelBase();
    GaussianModelBase(const std::vector<double> &y);
    GaussianModelBase * clone()const override =0;
    virtual double mu()const=0;
    virtual double sigsq()const=0;
    virtual double sigma()const;

    virtual void set_sigsq(double sigsq)=0;
    double pdf(Ptr<Data> dp, bool logscale)const override;
    double pdf(const Data * dp, bool logscale)const override;
    double Logp(double x, double &g, double &h, uint nd)const override;
    double Logp(const Vector & x, Vector &g, Matrix &h, uint nd)const;

    double ybar()const;
    double sample_var()const;

    void add_mixture_data(Ptr<Data>, double prob) override;
    double sim()const override;

    void add_data_raw(double x);
  };

}  // namespace BOOM
#endif// BOOM_GAUSSIAN_MODEL_BASE_HPP
