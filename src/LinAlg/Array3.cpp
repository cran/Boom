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
#include <LinAlg/Array3.hpp>
#include <LinAlg/Matrix.hpp>

namespace BOOM{
    typedef DeprecatedArray3 A3;
    typedef std::vector<double> dvector;

    A3::DeprecatedArray3()
      : n1(0),
	n2(0),
	n3(0),
	V()
    {}

    A3::DeprecatedArray3(uint dim, double x)
      : n1(dim),
	n2(dim),
	n3(dim),
	V(n1, Matrix(n2,n3, x))
    {}

    A3::DeprecatedArray3(uint d1, uint d2, uint d3, double x)
      : n1(d1),
	n2(d2),
	n3(d3),
	V(n1, Matrix(d2, d3, x))
    {}

    A3::DeprecatedArray3(const A3 & rhs)
      : n1(rhs.n1),
	n2(rhs.n2),
	n3(rhs.n3),
	V(rhs.V)
    {}


    A3 & A3::operator=(double x){
      if(n1==0){
	n1=n2=n3=1;
	V.resize(n1);
	V[0] = Matrix(1,1,x);
      }else std::fill(V.begin(), V.end(), x);
      return *this;
    }

    A3 & A3::operator=(const A3 &rhs){
      if(&rhs!=this){
	n1 = rhs.n1;
	n2 = rhs.n2;
	n3 = rhs.n3;
	V=rhs.V;
      }
      return *this;
    }

    bool A3::same_dim(const A3 &rhs)const{
      return (n1 == rhs.n1 &&
	      n2 == rhs.n2 &&
	      n3 == rhs.n3);}

    Matrix & A3::operator[](uint i){
      return V[i];}
    const Matrix & A3::operator[](uint i)const{
      return V[i];}

} // namespace BOOM
