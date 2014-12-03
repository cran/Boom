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

#ifndef DIRICHLET_MODEL_H
#define DIRICHLET_MODEL_H

#include <Models/ModelTypes.hpp>
#include <Models/VectorModel.hpp>

#include <Models/EmMixtureComponent.hpp>
#include <Models/ParamTypes.hpp>
#include <Models/Sufstat.hpp>
#include <Models/Policies/SufstatDataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/Policies/ParamPolicy_1.hpp>

namespace BOOM{

  class DirichletSuf : public SufstatDetails<VectorData>
  {
    Vec sumlog_;  // sum_i(log pi_j);
    double n_;
  public:
    // constructor
    DirichletSuf(uint S);
    DirichletSuf(const DirichletSuf &sf);
    DirichletSuf *clone() const;

    void clear();
    void Update(const VectorData &x);
    void add_mixture_data(const Vec &x, double prob);

    const Vec & sumlog()const;
    double n()const;
    DirichletSuf * abstract_combine(Sufstat *s);
    void combine(const DirichletSuf &);
    void combine(Ptr<DirichletSuf>);

    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true);
    virtual ostream & print(ostream &out)const;
  };

  //======================================================================
  class DirichletModel :
    public ParamPolicy_1<VectorParams>,
    public SufstatDataPolicy<VectorData, DirichletSuf>,
    public PriorPolicy,
    public DiffVectorModel,
    public NumOptModel,
    public EmMixtureComponent
  {
  public:
    DirichletModel(uint S);
    DirichletModel(uint S, double Nu);
    DirichletModel(const Vec &Nu);
    DirichletModel(const DirichletModel &m);
    DirichletModel *clone() const;

    Ptr<VectorParams> Nu();
    const Ptr<VectorParams> Nu()const;

    uint size()const;
    const Vec &nu() const;
    const double & nu(uint i)const;
    void set_nu(const Vec &);

    Vec pi()const;
    double pi(uint i)const;

    double pdf(dPtr dp, bool logscale) const;
    double pdf(const Data *, bool logscale) const;
    double pdf(const Vec &pi, bool logscale) const;
    double Logp(const Vec &p, Vec &g, Mat &h, uint lev) const ;
    double Loglike(const Vector &nu, Vec &g, Mat &h, uint nderiv) const;
    virtual void mle() {return d2LoglikeModel::mle();}

    double nu_loglike(const Vec & nu)const;

    Vec sim() const;
    virtual void add_mixture_data(Ptr<Data>, double prob);

  private:

  };

  //======================================================================

}
#endif
