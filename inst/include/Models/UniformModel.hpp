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
#ifndef BOOM_UNIFORM_MODEL_HPP
#define BOOM_UNIFORM_MODEL_HPP

#include <Models/ModelTypes.hpp>
#include <Models/DoubleModel.hpp>
#include <Models/Policies/ParamPolicy_2.hpp>
#include <Models/Policies/SufstatDataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/Sufstat.hpp>
#include <vector>

namespace BOOM{

  class UniformSuf
    : public SufstatDetails<DoubleData>
  {
  public:
    UniformSuf();
    UniformSuf(const std::vector<double> &d);
    UniformSuf(double low, double high);
    UniformSuf(const UniformSuf &rhs);
    UniformSuf * clone()const;

    void clear();
    void Update(const DoubleData &d);
    void update_raw(double x);

    double lo()const;
    double hi()const;

    void set_lo(double a);
    void set_hi(double b);
    void combine(Ptr<UniformSuf>);
    void combine(const UniformSuf &);
    UniformSuf * abstract_combine(Sufstat *s);

    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
                                            bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
                                            bool minimal=true);
    virtual ostream &print(ostream &out)const;
  private:
    double lo_, hi_;
  };

  class UniformModel
    : public ParamPolicy_2<UnivParams,UnivParams>,
      public SufstatDataPolicy<DoubleData, UniformSuf>,
      public PriorPolicy,
      public DiffDoubleModel,
      public LoglikeModel
  {
  public:
    UniformModel(double a=0, double b=1);
    UniformModel(const std::vector<double> & data);
    UniformModel(const UniformModel&rhs);
    UniformModel * clone()const;

    double lo()const;
    double hi()const;
    double nc()const;  // 1.0/(hi - lo);
    void set_lo(double a);
    void set_hi(double b);
    void set_ab(double a,double b);

    Ptr<UnivParams> LoParam();
    Ptr<UnivParams> HiParam();
    const Ptr<UnivParams> LoParam()const;
    const Ptr<UnivParams> HiParam()const;
    virtual double Logp(double x, double &g, double &h, uint nd)const;
    virtual double loglike()const;
    virtual void mle();
    virtual double sim()const;
  };
}
#endif  // BOOM_UNIFORM_MODEL_HPP;
