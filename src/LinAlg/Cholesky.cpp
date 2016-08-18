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
#include <LinAlg/Cholesky.hpp>
#include <cpputil/report_error.hpp>
#include <sstream>
#include <LinAlg/Vector.hpp>

extern "C"{
  /*  DPOTRF computes the Cholesky factorization of a real symmetric
   *  positive definite matrix A.
   */
  void dpotrf_(const char *, int *, double *, int *, int *);

  /*  DPOTRS solves a system of linear equations A*X = B with a symmetric
   *  positive definite matrix A using the Cholesky factorization
   *  A = U**T*U or A = L*L**T computed by DPOTRF.
   */
  void dpotrs_(const char *, int *, int *, const double *, int *, double *,
               int *, int*);

  /*  DPOTRI computes the inverse of a real symmetric positive definite
   *  matrix A using the Cholesky factorization A = U**T*U or A = L*L**T
   *  computed by DPOTRF.
   */
  void dpotri_(const char *, int *, double *, int *, int *);
}

namespace BOOM{
    Chol::Chol(const Matrix &m)
      : dcmp(m),
	pos_def(true)
    {
      if(!m.is_square()){
	pos_def=false;
	dcmp = Matrix();
      } else{
	int info=0;
	int n = m.nrow();
	dpotrf_("L", &n, dcmp.data(), &n, &info);
	if(info>0) pos_def=false;
      }
    }

    SpdMatrix Chol::original_matrix()const{
      return LLT(dcmp);
    }

    SpdMatrix Chol::inv()const{
      int n = dcmp.nrow();
      SpdMatrix ans(dcmp.begin(), dcmp.end());
      int info=0;
      dpotri_("L", &n, ans.data(), &n, &info);
      for(int i=0; i<n; ++i){
	for(int j=0; j<i; ++j){
	  ans(j,i) = ans(i,j);}}
      return ans;
    }

    uint Chol::nrow()const{ return dcmp.nrow();}
    uint Chol::ncol()const{ return dcmp.ncol();}
    uint Chol::dim()const{ return dcmp.nrow();}

    Chol & Chol::operator*=(double a){
      dcmp *= a;
      return *this;
    }

    Matrix Chol::getL()const{
      check();
      Matrix ans(dcmp);
      uint n = ans.nrow();
      for(uint i=1; i<n; ++i)
	std::fill( ans.col_begin(i), ans.col_begin(i) + i, 0.0);
      return ans;
    }

    Matrix Chol::getLT()const{
      check();
      Matrix ans(dcmp.t());
      uint n = ans.ncol();
      for(uint i = 1; i <n; ++i){
        VectorViewIterator b(ans.row_begin(i));
        std::fill(b, b+i, 0);
      }
      return(ans);
    }

    Matrix Chol::solve(const Matrix &B)const{
      check();
      Matrix ans(B);
      int n = dcmp.nrow();
      int ncol_b = B.ncol();
      int info=0;
      dpotrs_("L", &n, &ncol_b, dcmp.data(), &n, ans.data(), &n, &info);
      if(info<0){
	report_error("Chol::solve problem with cholesky solver");
      }
      return ans;
    }

    Vector Chol::solve(const Vector &B)const{

      // if *this is the cholesky decomposition of A then
      // this->solve(B) = A^{-1} B.  It is NOT L^{-1} B
      check();
      Vector ans(B);
      int n = dcmp.nrow();
      int ncol_b = 1;
      int info=0;
      dpotrs_("L", &n, &ncol_b, dcmp.data(), &n, ans.data(), &n, &info);
      if(info<0){
	report_error("Chol::solve problem with cholesky solver");
      }
      return ans;
    }

    // returns the log of the determinant of A
    double Chol::logdet()const{
      ConstVectorView d(diag(dcmp));
      double ans = 0;
      for(int i = 0; i < d.size(); ++i){
        ans += std::log(fabs(d[i]));
      }
      return 2 * ans;
    }

    double Chol::det()const{
      ConstVectorView d(diag(dcmp));
      double ans = d.prod();
      return ans * ans;
    }

    void Chol::check()const{
      if(!pos_def){
        std::ostringstream err;
        err << "attempt to use an invalid cholesky decomposition" << std::endl
            << "dcmp = " << std::endl
            << dcmp << std::endl
            << "original matrix = " << std::endl
            << original_matrix();
        report_error(err.str());
      }
    }

    Chol operator*(double a, const Chol &C){
      Chol ans(C);
      ans *= a;
      return ans;
    }

    Chol operator*(const Chol &C, double a){
      Chol ans(C);
      ans *= a;
      return ans;
    }
}
