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
#ifndef BOOM_NEWLA_ARRAY3_HPP
#define BOOM_NEWLA_ARRAY3_HPP

#include <cassert>
#include <cmath>
#include <iterator>
#include <vector>
#include <LinAlg/Matrix.hpp>
#include <boost/shared_ptr.hpp>

namespace BOOM{
    class DeprecatedArray3{
      uint n1, n2, n3;
      typedef std::vector<Matrix> mvector;
      mvector V;
    public:
      ~DeprecatedArray3(){}
      DeprecatedArray3();
      explicit DeprecatedArray3(uint dim, double x=0.0);
      DeprecatedArray3(uint d1, uint d2, uint d3, double x = 0.0);
      template <class Fwd>
      DeprecatedArray3(uint d1, uint d2, uint d3, Fwd Beg, Fwd End);

      DeprecatedArray3(const DeprecatedArray3 & rhs);

      DeprecatedArray3 & operator=(const DeprecatedArray3 &x);
      DeprecatedArray3 & operator=(double x);
      bool same_dim(const DeprecatedArray3 &x) const;

      uint dim1()const{return n1;}
      uint dim2()const{return n2;}
      uint dim3()const{return n3;}

      const Matrix &  operator[](uint i)const;
      Matrix &  operator[](uint i);

      inline double &  operator()(uint i, uint j, uint k);
      inline const double & operator()(uint i, uint j, uint k)const;
    };

    template <class Fwd>
    DeprecatedArray3::DeprecatedArray3(
        uint d1, uint d2, uint d3, Fwd Beg, Fwd End)
      : n1(d1),n2(d2),n3(d3),V(d1, Matrix(d2,d3))
    {
      uint d = std::distance(Beg,End);
      if(d== n1*n2*n3){
	for(uint i=0; i<n1; ++i)
	  Beg = std::copy(Beg, Beg+n2*n3, V[i].begin());
      }else if(d==n1){
	std::copy(Beg,End, V.begin());
      }else{
        assert(false && "invalid sequence in DeprecatedArray3 constructor");
      }
    }

    inline double & DeprecatedArray3::operator()(uint i, uint j, uint k){
      return V[i](j,k);}

    inline const double & DeprecatedArray3::operator()(
        uint i, uint j, uint k) const {return V[i](j,k);}

}
#endif// BOOM_NEWLA_ARRAY3_HPP
