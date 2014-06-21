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

#include <TargetFun/TargetFun.hpp>
#include <cmath>
#include <LinAlg/Matrix.hpp>
// #include <LinAlg/Types.hpp>

namespace BOOM{


  void intrusive_ptr_add_ref(TargetFun *s){
    s->up_count();}
  void intrusive_ptr_release(TargetFun *s){
    s->down_count();
    if(s->ref_count()==0) delete s; }

  dTargetFun::dTargetFun() : eps_scale(1e-5){}

//   Mat dTargetFun::h_approx(const Vec &x)const{
//     uint k = x.size();
//     Mat ans(k,k);

//     Vec y = x;
//     Vec df1 = x;
//     Vec df2 = x;
//     for(uint i=0; i<k; ++i){
//       double eps = (y[i]==0 ? eps_scale : eps_scale * fabs(y[i]));
//       y[i]+=eps;
//       d1(y,df1);
//       y[i]-= eps;
//       d1(y,df2);
//       for(uint j=0; j<k; ++j) ans(i,j) = (df1[j]-df2[j])/(2*eps);
//       y[i]=x[i];
//     }
//     return ans;
//   }

  //======================================================================
  void intrusive_ptr_add_ref(ScalarTargetFun *s){
    s->up_count();}
  void intrusive_ptr_release(ScalarTargetFun *s){
    s->down_count();
    if(s->ref_count()==0) delete s; }
  //----------------------------------------------------------------------
  dScalarTargetFun::dScalarTargetFun() : eps_scale(1e-5){}

//   double dScalarTargetFun::h_approx(const double &x)const{
//     double y(x), df1(0), df2(0);
//     double eps = (y==0 ? eps_scale : eps_scale * fabs(y));
//     y = x+eps;
//     d1(y,df1);
//     y= x-eps;
//     d1(y,df2);
//     return (df1 - df2)/(2*eps);
//   }

  //======================================================================

  ScalarTargetView::ScalarTargetView(TargetFun &F, const Vec &X, uint which_dim)
    : f(F),
      wsp(X),
      which(which_dim)
  {}

  double ScalarTargetView::operator()(double x)const{
    wsp[which] = x;
    return f(wsp);
  }

  void ScalarTargetView::set_x(const Vec &X){ wsp = X;}
}
