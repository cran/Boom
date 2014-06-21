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
    GaussianSuf(double S, double Ssq, double N);
    GaussianSuf(const GaussianSuf &);
    GaussianSuf *clone() const;

    void clear();
    void Update(const DoubleData &X);
    void update_raw(double y);

    // Remove the effect of observation y from the sufficient
    // statistics, as if it were dropped from the data set.
    void remove(double y);
    void add_mixture_data(double y, double prob);
    double sum()const;
    // sumsq is the uncentered (raw) sum of squared y's.
    double sumsq()const;
    double n()const;

    double ybar()const;
    double sample_var()const;

    GaussianSuf * abstract_combine(Sufstat *s);
    void combine(Ptr<GaussianSuf>);
    void combine(const GaussianSuf &);
    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true);
    virtual ostream &print(ostream &out)const;
  };
  //======================================================================
  class GaussianModelBase
      : public SufstatDataPolicy<DoubleData, GaussianSuf>,
        public DiffDoubleModel,    // promises  Logp(x,g,h,nd);
        public NumOptModel,        // promises Loglike(g,h,nd), and mle();
        public EmMixtureComponent  // promises add_mixture_data,
  {                                //   and find_posterior_mode()
   public:
    GaussianModelBase();
    GaussianModelBase(const std::vector<double> &y);
    virtual GaussianModelBase * clone()const=0;
    virtual double mu()const=0;
    virtual double sigsq()const=0;
    virtual double sigma()const;

    virtual void set_sigsq(double sigsq)=0;
    virtual double pdf(Ptr<Data> dp, bool logscale)const;
    virtual double pdf(const Data * dp, bool logscale)const;
    double Logp(double x, double &g, double &h, uint nd)const;
    double Logp(const Vec & x, Vec &g, Mat &h, uint nd)const;

    double ybar()const;
    double sample_var()const;

    virtual void add_mixture_data(Ptr<Data>, double prob);
    virtual double sim()const;

    void add_data_raw(double x);
  };

}
#endif// BOOM_GAUSSIAN_MODEL_BASE_HPP
