/*
  Copyright (C) 2005-2009 Steven L. Scott

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
#ifndef BOOM_PRODUCT_DIRICHLET_MODEL_HPP
#define BOOM_PRODUCT_DIRICHLET_MODEL_HPP
#include <Models/ModelTypes.hpp>
#include <Models/ParamTypes.hpp>
#include <Models/Sufstat.hpp>
#include <Models/Policies/SufstatDataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/Policies/ParamPolicy_1.hpp>

namespace BOOM{

  class ProductDirichletSuf
    : public SufstatDetails<MatrixData>
  {
  public:
    ProductDirichletSuf(uint p);
    ProductDirichletSuf(const ProductDirichletSuf &rhs);
    ProductDirichletSuf * clone()const;
    const Mat & sumlog()const;
    double n()const;
    void Update(const MatrixData &);
    void clear();
    void combine(Ptr<ProductDirichletSuf>);
    void combine(const ProductDirichletSuf &);
    ProductDirichletSuf * abstract_combine(Sufstat *s);

    virtual Vec vectorize(bool minimal=true)const;
    virtual Vec::const_iterator unvectorize(Vec::const_iterator &v,
					    bool minimal=true);
    virtual Vec::const_iterator unvectorize(const Vec &v,
					    bool minimal=true);
    virtual ostream &print(ostream &out)const;
  private:
    Mat sumlog_;
    double n_;
  };


// class ProductDirichletModelBase
//     : public SufstatDataPolicy<MatrixData, ProductDirichletSuf>



  class ProductDirichletModel
    : public ParamPolicy_1<MatrixParams>,
      public SufstatDataPolicy<MatrixData, ProductDirichletSuf>,
      public PriorPolicy,
      public dLoglikeModel
  {
  public:
    ProductDirichletModel(uint p);  // default is uniform:  Nu = 1
    ProductDirichletModel(const Mat &NU);
    ProductDirichletModel(const Vec &wgt, const Mat &Pi);
    ProductDirichletModel(const ProductDirichletModel &);

    ProductDirichletModel * clone()const;
    uint dim()const;
    Ptr<MatrixParams> Nu_prm();
    const Ptr<MatrixParams> Nu_prm()const;
    const Mat & Nu()const;

    void set_Nu(const Mat &Nu);

    double pdf(Ptr<Data>, bool logscale)const;
    double pdf(const Mat &Pi, bool logscale)const;
    //    double Logp(const Vec &, Vec &, Mat &, uint nd)const;

    double loglike()const;
    double dloglike(Vec &g)const;

    Mat sim()const;
  };
}

#endif // BOOM_PRODUCT_DIRICHLET_MODEL_HPP
