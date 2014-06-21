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

#ifndef POISSON_MODEL_H
#define POISSON_MODEL_H

#include <Models/ModelTypes.hpp>
#include <Models/Sufstat.hpp>
#include <Models/Policies/ParamPolicy_1.hpp>
#include <Models/Policies/SufstatDataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>

//----------------------------------------------------------------------//
namespace BOOM{

  class PoissonSuf : public SufstatDetails<IntData>{
  public:
    // constructor
    PoissonSuf();

    // If this constructor is used, then the normalizing constant will
    // not be correctly set.  That's probably okay for most
    // applications.
    PoissonSuf(double event_count, double exposure);

    PoissonSuf(const PoissonSuf &rhs);
    PoissonSuf *clone() const;

    // If this function is used to set the value of the sufficient
    // statistics, then the normalizing constant will be wrong, which
    // is probably fine for most applications.
    void set(double event_count, double exposure);

    void clear();
    double sum()const;
    double n()const;
    double lognc()const;

    void Update(const IntData & dat);
    void add_mixture_data(double y, double prob);
    void combine(Ptr<PoissonSuf>);
    void combine(const PoissonSuf &);
    PoissonSuf * abstract_combine(Sufstat *s);

    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true);
    virtual ostream &print(ostream &out)const;

  private:
    double sum_, n_, lognc_;  // log nc is the log product of x-factorials
  };
  //----------------------------------------------------------------------//

  class PoissonModel : public ParamPolicy_1<UnivParams>,
		       public SufstatDataPolicy<IntData, PoissonSuf>,
		       public PriorPolicy,
		       public NumOptModel,
		       public MixtureComponent
  {
  public:

    PoissonModel(double lam=1.0);
    PoissonModel(const std::vector<uint> &);
    PoissonModel(const PoissonModel &m);
    PoissonModel *clone() const;

    virtual void mle();
    virtual double Loglike(Vec &g, Mat &h, uint nd)const;

    Ptr<UnivParams> Lam();
    const Ptr<UnivParams> Lam()const;
    double lam()const;
    void set_lam(double);

    // probability calculations
    virtual double pdf(Ptr<Data> x, bool logscale) const;
    virtual double pdf(const Data * x, bool logscale) const;
    double pdf(uint x, bool logscale) const;

    // moments and summaries:
    double mean()const;
    double var()const;
    double sd()const;
    double simdat() const;

    virtual void add_mixture_data(Ptr<Data>,  double prob);
  };


}
#endif
