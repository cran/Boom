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

#include <LinAlg/Types.hpp>
#include <LinAlg/Matrix.hpp> // includes Vector.hpp as well
#include <LinAlg/Vector.hpp>
#include <numopt.hpp>
#include <sstream>
#include <iostream>

namespace BOOM{
  using std::endl;

  namespace{
    inline bool BAD(double lcrit, double epsilon){
      return (lcrit!=lcrit)            // nan
          || ( (fabs(lcrit) > epsilon) && (lcrit < 0 ) )  //
          ;
    }

    inline bool keep_going(double lcrit, double leps, int nit,
                           int nitmax, int sh){
      if(sh>0) return true;   // last iteration used step halving.. not done.
      else if(nit >= nitmax) return false; // max iterations exceeded
      else if(lcrit > leps) return true;   // not there yet.
      return false;                        // mission accomplished!
    }
  }
  // ======================================================================*/
  // Newton-Raphson routine to MINIMIZE the target function tf.
  // If you want to maximize tf consider calling max_nd2 instead.
  // theta is the initial value, g and h will be returned.  leps is
  // an epsilon considering changes in the function value.  The
  // 'target function' tf takes theta as its first argument and
  // returns g and h in its second and third arguments.
  //
  // This function implements step_halving if an increase in the
  // function value occurs.
  //
  // The total number of function calls is returned in 'fc'.
  double newton_raphson_min(Vec &theta, Vec &g, Mat &h, d2Target tf,
                            int &fc, double leps, bool & happy_ending,
                            string &error_message){
    double loglike=0, oldloglike, lcrit=1+leps;;
    int nriter=0, nritermax=30, sh=0, total_sh=0, maxsh=10, maxtsh=50;

    fc=0;
    happy_ending=true;
    error_message = "";
    oldloglike=tf(theta, g, h); ++fc;
    while( keep_going(lcrit, leps, nriter, nritermax, sh) ){
      ++nriter;
      Vec step = h.solve(g);
      theta -= step;
      double directional_derivative = g.dot(step);
      loglike=tf(theta, g, h); ++fc;
      lcrit=oldloglike-loglike;    // should be positive if all is well
      sh=0;
      if(BAD(lcrit, leps/2) ){ /* step halving */
        if(directional_derivative < 0){
          // mathematically this is impossible, because step =
          // -H.inv() * g so the directional derivative is -g*Hinv*g,
          // which is must be negative.  If you get here, please check
          // that you have defined your target function correctly
          if(fabs(directional_derivative) < leps) return loglike;
        }
 	++total_sh;
 	Vec oldtheta=theta+step;
 	double  alpha=1.0;
 	while(BAD(lcrit, leps/2) && (sh <= maxsh) ){
 	  ++sh;
 	  alpha/=2.0;
          step *= alpha;            // halve step size
 	  theta = oldtheta - step;
 	  loglike = tf(theta, g, h); ++ fc;
 	  lcrit = oldloglike-loglike;
        }
 	if(!h.is_pos_def()){
          happy_ending = false;
          ostringstream err;
          err << "The Hessian matrix is not positive definite in newton_raphson_min"
              << endl << h << endl;
          error_message = err.str();
          return loglike;
        }
      }

      oldloglike = loglike;
      if((sh>maxsh) || (total_sh > maxtsh)){
        happy_ending=false;
        return loglike;
      }
    }
    return loglike;
  }
  /*======================================================================*/
}
