/*
  Copyright (C) 2012 Steven L. Scott

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

#ifndef BOOM_INDEPENDENT_MVN_MODEL_HPP
#define BOOM_INDEPENDENT_MVN_MODEL_HPP

#include <Models/MvnBase.hpp>
#include <Models/Policies/ParamPolicy_2.hpp>
#include <Models/Policies/SufstatDataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>

namespace BOOM{
  class IndependentMvnSuf : public SufstatDetails<VectorData> {
   public:
    IndependentMvnSuf(int dim);
    IndependentMvnSuf * clone()const;

    void clear();
    void resize(int dim);
    void Update(const VectorData &);
    void update_raw(const Vec &x);
    void add_mixture_data(const Vec &x, double prob);

    double sum(int i)const;
    double sumsq(int i)const;  // uncentered sum of squares
    double n(int i)const;

    double ybar(int i)const;
    double sample_var(int i)const;

    IndependentMvnSuf * abstract_combine(Sufstat *s);
    void combine(Ptr<IndependentMvnSuf>);
    void combine(const IndependentMvnSuf &);
    virtual Vec vectorize(bool minimal = true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
                                            bool minimal = true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
                                            bool minimal = true);
    virtual ostream & print(ostream &out)const;
   private:
    Vec sum_;
    Vec sumsq_;
    Vec n_;
  };

  class IndependentMvnModel
      : public MvnBase,
        public ParamPolicy_2<VectorParams, VectorParams>,
        public SufstatDataPolicy<VectorData, IndependentMvnSuf>,
        public PriorPolicy,
        virtual public MixtureComponent
  {
  public:
    IndependentMvnModel(int dim);
    IndependentMvnModel(const Vector &mean,
                        const Vector &variance);
    IndependentMvnModel(const IndependentMvnModel &rhs);
    virtual IndependentMvnModel * clone()const;
    // Several virtual functions from MvnBase are re-implemented here
    // for efficiency.
    virtual double Logp(const Vec &x, Vec &g, Mat &h, uint nderiv)const;
    virtual const Vec & mu() const;
    virtual const Spd & Sigma()const;
    virtual const Spd & siginv() const;
    virtual double ldsi()const;
    virtual Vec sim()const;

    Ptr<VectorParams> Mu_prm();
    const Ptr<VectorParams> Mu_prm()const;
    const VectorParams & Mu_ref()const;

    Ptr<VectorParams> Sigsq_prm();
    const Ptr<VectorParams> Sigsq_prm()const;
    const VectorParams & Sigsq_ref()const;

    const Vec &sigsq()const;
    double mu(int i)const;
    double sigsq(int i)const;
    double sigma(int i)const;

    void set_mu(const Vec &mu);
    void set_mu_element(double mu, int position);
    void set_sigsq(const Vec &sigsq);
    void set_sigsq_element(double sigsq, int position);

    virtual double pdf(const Data * dp, bool logscale)const;
   private:
    mutable Spd sigma_scratch_;
    mutable Vec g_;
    mutable Mat h_;
  };

}
#endif //  BOOM_INDEPENDENT_MVN_MODEL_HPP
