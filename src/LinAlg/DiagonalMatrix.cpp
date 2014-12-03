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
#include <LinAlg/DiagonalMatrix.hpp>
#include <LinAlg/Matrix.hpp>
#include <LinAlg/SpdMatrix.hpp>
#include <LinAlg/blas.hpp>

#include <distributions.hpp>
#include <algorithm>
#include <numeric>
#include <functional>

namespace BOOM{
    using namespace blas;
    typedef DiagonalMatrix DM;

    DM::DiagonalMatrix()
      : Matrix()
    {}
    DM::DiagonalMatrix(uint n, double x)
      : Matrix(n,n)
    {
      diag()=x; // set using VectorView
    }

    DM::DiagonalMatrix(const DiagonalMatrix &rhs)
      : Matrix(rhs)
    {}

    DM::DiagonalMatrix(const Matrix &rhs)
      : Matrix(rhs.nrow(), rhs.ncol())
    {
      assert(rhs.is_square());
      diag()=rhs.diag();
    }

    DiagonalMatrix & DM::operator=(const DiagonalMatrix &rhs){
      if(&rhs!=this) Matrix::operator=(rhs);
      return *this; }

    DiagonalMatrix & DM::operator=(const Matrix &rhs){
      assert(rhs.is_square());
      Matrix::resize(rhs.nrow(), rhs.ncol());
      set_diag(rhs.diag());
      return *this;
    }

    DiagonalMatrix & DM::operator=(const double &x){
      if(nrow()==0) Matrix::resize(1,1);
      diag()=x;
      return *this;
    }

    bool DM::operator==(const DiagonalMatrix &rhs)const{
      if(!same_dim(rhs)) return false;
      return std::equal(dbegin(), dend(), rhs.dbegin());
    }

    void DM::swap(DM &rhs){ Matrix::swap(rhs);}

    void DM::randomize(){
      uint n = nrow();
      VectorView d(diag());
      for(uint i=0; i<n; ++i) d[i] = runif(0,1);
    }

    DM & DM::resize(uint n){
      if(nrow()!=n){
        Vector d(diag());
        d.resize(n);
        Matrix::resize(n,n);
        set_diag(d);
      }
      return *this;
    }

    double & DM::operator[](uint n){
      assert(inrange(n,n));
      return unchecked(n,n); }

    const double & DM::operator[](uint n)const{
      assert(inrange(n,n));
      return unchecked(n,n); }

    //---------------  Matrix multiplication -----------
    class scale_mult{  // useful functor for STL multiplication
      double x;
    public:
      scale_mult(double d) : x(d){}
      double operator()(double a, double b)const{return x*a*b;}
    };

    Matrix & DM::mult(const Matrix &B, Matrix & ans, double scal)const{
      // scale the column of
      assert(nrow()==ans.nrow() && ncol()==B.nrow() && B.ncol()==ans.ncol());
      for(uint i=0; i<ncol(); ++i){
        double a = unchecked(i,i)*scal;
        ConstVectorView b(B.row(i));
        VectorView Ans(ans.row(i));
        daxpy(b.size(), a, b.data(), b.stride(), Ans.data(), Ans.stride());}
      return ans;}

    Matrix & DM::Tmult(const Matrix &B, Matrix & ans, double scal)const{
      return this->mult(B,ans, scal);}

    Matrix & DM::multT(const Matrix &B, Matrix & ans, double scal)const{
      assert(nrow()==ans.nrow() && B.nrow()==ans.ncol() && ncol()==B.ncol());
      for(uint i=0; i<nrow(); ++i){
        double a = unchecked(i,i)*scal;
        VectorView b(B.col(i));
        VectorView Ans(ans.row(i));
        daxpy(b.size(), a, b.data(), b.stride(), Ans.data(), Ans.stride());}
      return ans;}

    //------ SpdMatrix (this and spd both symmetric) ----------

    Matrix & DM::mult(const SpdMatrix &S, Matrix & ans, double scal)const{
      const Matrix &tmp(S);
      return this->mult(tmp, ans, scal);}

    Matrix & DM::Tmult(const SpdMatrix &S, Matrix & ans, double scal)const{
      const Matrix &tmp(S);
      return this->mult(tmp, ans, scal);}

    Matrix & DM::multT(const SpdMatrix &S, Matrix & ans, double scal)const{
      const Matrix &tmp(S);
      return this->mult(tmp, ans, scal);}

    //------ DiagonalMatrix (this and spd both symmetric) ----------

    DiagonalMatrix & DM::mult(const DiagonalMatrix &S, Matrix & ans,
                              double scal)const{
      DiagonalMatrix &D(dynamic_cast<DiagonalMatrix &>(ans));
      assert(can_mult(S,ans));
      if(scal==1.0)
        std::transform(dbegin(), dend(), S.dbegin(), D.dbegin(),
                       std::multiplies<double>());
      else
        std::transform(dbegin(), dend(), S.dbegin(), D.dbegin(),
                       scale_mult(scal));
      return D;
    }

    DiagonalMatrix & DM::Tmult(const DiagonalMatrix &S, Matrix & ans,
                               double scal)const{
      return mult(S,ans, scal);}

    DiagonalMatrix & DM::multT(const DiagonalMatrix &S, Matrix & ans,
                               double scal)const{
      return mult(S,ans, scal);}

    //---------- Vector ------------
    Vector & DM::mult(const Vector &v, Vector &ans, double scal)const{
      assert(v.size()==ans.size());
      if(scal==1.0)
        std::transform(dbegin(), dend(), v.begin(), ans.begin(),
                       std::multiplies<double>());
      else
        std::transform(dbegin(), dend(), v.begin(), ans.begin(),
                       scale_mult(scal));
      return ans;
    }

    Vector & DM::Tmult(const Vector &v, Vector &ans, double scal)const{
      return this->mult(v,ans, scal);}

    DiagonalMatrix DM::t()const{ return *this;}

    DiagonalMatrix DM::inv()const{
      DiagonalMatrix ans(nrow());
      VectorView d(ans.diag());
      ConstVectorView cd(diag());
      for(uint i=0; i<nrow(); ++i){
        d[i] = 1.0/cd[i];
      }
      return ans;
    }

    SpdMatrix DM::inner()const{
      SpdMatrix ans(nrow());
      std::transform(dbegin(), dend(), dbegin(), ans.dbegin(),
                     std::multiplies<double>());
      return ans;
    }

    Matrix DM::solve(const Matrix &mat)const{
      assert(ncol()==mat.nrow());
      Matrix ans(mat.nrow(), mat.ncol());
      for(uint i=0; i<nrow(); ++i){
        double a = unchecked(i,i);
        ConstVectorView b(mat.row(i));
        VectorView Ans(ans.row(i));
        daxpy(b.size(), a, b.data(), b.stride(), Ans.data(), Ans.stride());}
      return ans;}

    Vector DM::solve(const Vector &v)const{
      assert(nrow()==v.size());
      Vector ans(v.size());
      std::transform(v.begin(), v.end(), dbegin(), ans.begin(),
                     std::divides<double>());
      return ans;
    }

    double DM::det()const{return prod();}
    Vector DM::singular_values()const{
      Vector ans(diag());
      std::sort(ans.begin(), ans.end(), std::greater<double>());
      return ans;
    }

    Vector DM::real_evals()const{
      Vector ans(diag());
      std::sort(ans.begin(), ans.end(), std::greater<double>());
      return ans;
    }


    DM & DM::operator+=(double x){
      diag()+=x;
      return *this; }
    DM & DM::operator-=(double x){
      diag()-=x;
      return *this; }
    DM & DM::operator*=(double x){
      diag()*=x;
      return *this; }
    DM & DM::operator/=(double x){
      diag()/=x;
      return *this; }


    double DM::sum()const{
      return std::accumulate(dbegin(), dend(), 0.0);}

    double DM::prod()const{
      return std::accumulate(dbegin(), dend(), 1.0,
                             std::multiplies<double>());}
}
