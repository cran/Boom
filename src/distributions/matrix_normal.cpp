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
#include <cmath>

#include <distributions.hpp>

#include <LinAlg/Types.hpp>
#include <LinAlg/Vector.hpp>
#include <LinAlg/Matrix.hpp>
#include <LinAlg/Cholesky.hpp>
#include <LinAlg/SpdMatrix.hpp>
#include <algorithm>

#include <iostream>

namespace BOOM{

  double dmatrix_normal_ivar(const Mat &Y,const Mat &Mu,
			const Spd & Siginv, const Spd & Ominv,
			bool logscale){

    double ldoi = Ominv.logdet();
    double ldsi = Siginv.logdet();
    return dmatrix_normal_ivar(Y,Mu,Siginv, ldsi, Ominv, ldoi, logscale);
  }

  double dmatrix_normal_ivar(const Mat &Y,const Mat &Mu,
			const Spd & Siginv, double ldsi,
			const Spd & Ominv, double ldoi,
			bool logscale){

    // Y~ matrix_normal(Mu, Siginv, Ominv) if
    // Vec(Y) ~ N(Vec(Mu), (Siginv \otimes Ominv)^{-1})
    // Where dim(Y) = (xdim,ydim) , dim(Siginv) = ydim,
    // dim(Ominv) = xdim

    Mat E =  Y-Mu;
    Mat A = Ominv * E;
    Mat B = E * Siginv;
    double qform = traceAtB(A,B);

    // qform = vec(Y-Mu)^T (Siginv \otimes Ominv) vec(Y-Mu)
    //  = tr(E^T Ominv E Siginv)    see Harville (1997) 16.2.15

    uint xdim = Y.nrow();
    uint ydim = Y.ncol();
    double logdet_ivar = ydim * ldoi + xdim * ldsi;
    // xsize * ydeterm. + oppositex

    const double log2pi = 1.83787706641;

    uint n = xdim*ydim;
    double ans = -.5*n*log2pi + .5*logdet_ivar - .5*qform;
    return logscale ? ans : exp(ans);
  }

  Mat rmatrix_normal_ivar(const Mat & Mu, const Spd &Siginv, const Spd &Ominv){
    return rmatrix_normal_ivar_mt(GlobalRng::rng, Mu, Siginv, Ominv);}

  Mat rmatrix_normal_ivar_mt(RNG & rng, const Mat & Mu,
                             const Spd &Siginv, const Spd &Ominv){

    uint xdim = Mu.nrow();
    uint ydim = Mu.ncol();
    Mat Z(xdim,ydim);
    double *zdata = Z.data();
    for(uint i=0; i<xdim*ydim; ++i){ zdata[i] = rnorm_mt(rng); }

    Mat Ominv_U(t(Chol(Ominv).getL()));
    Mat Lsig(Linv(Chol(Siginv).getL()));

    Mat ans = Mu + Usolve(Ominv_U,Z) * Lsig;
    return ans;
  }

}
