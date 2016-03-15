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

#include <LinAlg/SWEEP.hpp>
#include <LinAlg/Types.hpp>

namespace BOOM{
  typedef SweptVarianceMatrix SVM;

  SVM::SweptVarianceMatrix()
      : S(), swept_(), nswept_(0) {}

  SVM::SweptVarianceMatrix(uint d)
      : S(d), swept_(d, false), nswept_(0){}

  SVM::SweptVarianceMatrix(const SpdMatrix &m)
      : S(m), swept_(m.nrow(), false), nswept_(0){}

  SVM::SweptVarianceMatrix
      (const SVM &rhs):
      S(rhs.S),
      swept_(rhs.swept_),
      nswept_(rhs.nswept_)
      {}

  void SVM::SWP(const std::vector<bool> &inc){
    uint p = inc.size();
    assert(p==S.nrow());
    for(uint i=0; i<p; ++i){
      if(inc[i]) SWP(i);
      else RSW(i);}}

  void SVM::SWP(uint m){
    if(swept_[m]) return;
    ++nswept_;
    swept_[m]=true;
    double x = S(m,m);
    uint d = S.dim();
    for(uint i=0; i<d; ++i){
      if(i!=m){
        for(uint j = 0; j<d; ++j){
          if(j!=m){
            S(i,j) -= S(i,m)*S(m,j)/x; }}}}
    S(m,m) = -1.0/x;
    for(uint i = 0; i<d; ++i){
      if(i!=m){
        S(i,m)/=x;
        S(m,i)/=x;}}
  }

  void SVM::RSW(uint m){
    if(!swept_[m]) return;
    --nswept_;
    swept_[m]=false;

    double x = S(m,m);
    uint d = S.dim();
    for(uint i=0; i<d; ++i){
      if(i!=m){
        for(uint j = 0; j<d; ++j){
          if(j!=m){
            S(i,j) -=  S(i,m)*S(m,j)/x; }}}}

    S(m,m) = -1.0/x;
    for(uint i = 0; i<d; ++i){
      if(i!=m){
        S(i,m)/=x;
        S(m,i)/=x;}}
  }

  uint SVM::xdim()const{ return nswept_; }
  uint SVM::ydim()const{ return S.nrow()- nswept_; }

  //------------------------------------------------------------


  Matrix SVM::Beta()const{   // E(y|x) = x*Beta
    Matrix ans(xdim(), ydim());
    uint ii=0;
    for(uint i = 0; i<S.dim(); ++i){
      if(swept_[i]==true){  // i is an 'x' dimension
        uint jj=0;
        for(uint j = 0; j<S.dim(); ++j){
          if(swept_[j]==false)     // j is a 'y' dimension
            ans(ii,jj++)=S(i,j);}
        if(jj==ydim()) break;
        ++ii;}
      if(ii==xdim()) break; }
    return ans; }

  Vector SVM::E_y_given_x(const Vector &x, const Vector &mu){
    assert(mu.size()==S.ncol());
    assert(x.size() == nswept_);
    std::vector<bool> isy(swept_);
    isy.flip();
    return x * Beta() + select(mu, isy);
  }
  //------------------------------------------------------------

  SpdMatrix SVM::V_y_given_x()const{
    SpdMatrix ans(ydim());
    uint ii=0;
    uint d = S.dim();
    for(uint i = 0; i<d; ++i){
      if(swept_[i]==true){
        uint jj=0;
        for(uint j  = 0; j<=i; ++j){
          if(swept_[j]==true){
            ans(ii,jj) = ans(jj,ii) = S(i,j);
            ++jj;}}
        ++ii;}}
    return ans;}

  SpdMatrix SVM::ivar_x()const{
    SpdMatrix ans(xdim());
    uint ii=0;
    uint d = S.dim();
    for(uint i = 0; i<d; ++i){
      if(swept_[i]==true){
        uint jj=0;
        for(uint j  = 0; j<=i; ++j){
          if(swept_[j]==true){
            ans(ii,jj) = ans(jj,ii) = -S(i,j);
            ++jj;}}
        ++ii;}}
    return ans;}

} // ends namespace BOOM
