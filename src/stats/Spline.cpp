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
#include <stats/Spline.hpp>
#include <cpputil/report_error.hpp>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <BOOM.hpp>

namespace BOOM{

  Spline::Spline(const Vector &Knots, uint ord)
    : order_(ord),
      ordm1_(ord-1),
      curs(0),
      boundary(0),
      knots(Knots),
      rdel(ord-1),
      ldel(ord-1),
      a(ord)
  {
    if(nknots()<=0){
      report_error("you must have at least one knot to use a Spline");
    }
    std::sort(knots.begin(), knots.end());
  }

  int Spline::nknots()const{
    return static_cast<int>(knots.size());
  }

  void Spline::set_cursor(double x)const{
    // do not assume xs are sorted
    curs = -1;          // Wall
    boundary = 0;
    for(int i=0; i<nknots(); ++i){
      if (knots[i] >= x) curs = i;
      if (knots[i] > x) break;
    }

    if (curs > nknots() - order_) {
      int lastLegit = nknots() - order_;
      if (x == knots[lastLegit]) {
	boundary = 1; curs = lastLegit;
      }
    }
  }


  const Vector & Spline::basis(double x, Vector &ans)const{
    set_cursor(x);
    int offsets = curs - order_;
    int j = offsets;
    int nk = nknots();
    ans.resize(order_);
    if (j < 0 || j > nk) {
      std::ostringstream err;
      err << "a bad bad thing happened in Spline::basis()" << endl
	  << "x = " << x << endl
	  << "j = " << j << endl
	  << "nk = " << nk << endl
	  << "curs = " << curs << endl
	  << "offsets = " << offsets << endl
	;
      report_error(err.str());
    } else {
      basis_funcs(x, ans);
    }
    return ans;
  }

  Vector Spline::basis(double x)const{
    Vector ans(order_);
    basis(x,ans);
    return ans;
  }

  void Spline::diff_table(double x, int ndiff)const
  {
    for (int i = 0; i < ndiff; i++) {
      rdel[i] = knots[curs + i] - x;
      ldel[i] = x - knots[curs - (i + 1)];
    }
  }


  /* fast evaluation of basis functions */
  void Spline::basis_funcs(double x, Vector &b)const
  {
    diff_table(x, ordm1_);
    b[0] = 1.;
    for (int j = 1; j <= ordm1_; j++) {
      double saved = 0.;
      for (int r = 0; r < j; r++) {
	double term = b[r]/(rdel[r] + ldel[j - 1 - r]);
	b[r] = saved + rdel[r] * term;
	saved = ldel[j - 1 - r] * term;
      }
      b[j] = saved;
    }
  }

  double Spline::eval(double x, const Vector &beta)const{
    set_cursor(x);
    double ans(0);
    if(curs < order_ || curs > (nknots() - order_) ){
      report_error("a bad bad thing happened in Spline::eval");
    } else {
      memcpy(a.data(), beta.data() + curs - order_, order_);
      ans = evaluate_derivs(x, 0);
    }
    return ans;
  }

  double Spline::evaluate_derivs(double x, int nder)const {

    const double *lpt, *rpt, *ti = knots.data() + curs;
    double *apt;
    int inner, outer = ordm1_;

    if (boundary && nder == ordm1_) { /* value is arbitrary */
      return 0.0;
    }
    while(nder--) {
      for(inner = outer, apt = a.data(), lpt = ti - outer; inner--; apt++, lpt++)
	*apt = outer * (*(apt + 1) - *apt)/(*(lpt + outer) - *lpt);
      outer--;
    }
    diff_table(x, outer);
    while(outer--)
      for(apt = a.data(), lpt = ldel.data() + outer, rpt = rdel.data(), inner = outer + 1;
	  inner--; lpt--, rpt++, apt++)
	*apt = (*(apt + 1) * *lpt + *apt * *rpt)/(*rpt + *lpt);
    return a[0];
  }



}
