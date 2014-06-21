/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#include <Models/Glm/BinomialLogitModel.hpp>
#include <stdexcept>
#include <cpputil/ThrowException.hpp>

namespace BOOM{
  typedef BinomialRegressionData BRD;
  BRD::BinomialRegressionData(uint y, uint n, const Vec &x, bool add_icpt)
      : GlmData<IntData>(y,x,add_icpt),
        n_(n)
  {
    check();
  }

  BRD * BRD::clone()const{ return new BRD(*this);}

  void BRD::set_n(uint n, bool check_n){
    n_ = n;
    if(check_n) check();
  }

  void BRD::set_y(uint y, bool check_n){
    GlmData<IntData>::set_y(y);
    if(check_n) check();
  }

  uint BRD::n()const{return n_;}

  void BRD::check()const{
    if( n_<y() ){
      ostringstream err;
      err << "error in BinomialRegressionData:  n < y" << endl
          << "  n = " << n_ << endl
          << "  y = " << y() << endl
          ;
      throw_exception<std::runtime_error>(err.str());
    }
  }

  ostream & BRD::display(ostream &out)const{
    out << n_ << " ";
    return GlmData<IntData>::display(out);
  }
}
