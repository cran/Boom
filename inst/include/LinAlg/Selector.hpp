/*
  Copyright (C) 2005-2009 Steven L. Scott

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

#ifndef BOOM_SELECTOR_HPP
#define BOOM_SELECTOR_HPP

#include <vector>
#include <iostream>
#include <BOOM.hpp>
#include <cassert>
#include <string>
#include <LinAlg/Types.hpp>
#include <distributions/rng.hpp>

namespace BOOM{

  //TODO(stevescott):  remove the inheritance from vector<bool>
  class Selector : public std::vector<bool> {
   public:
    typedef unsigned int uint;
   private:
    std::vector<uint> inc_indx; // sorted vector of included indices
    bool include_all;  //
    void reset_inc_indx();
    // Checks that the size of this is equal to 'p'.
    void check_size_eq(uint p, const string &fun_name)const;
    // Checks that the size of *this is greater than 'p'.
    void check_size_gt(uint p, const string &fun_name)const;
   public:
    Selector();
    explicit Selector(uint p, bool all=true);  // all true or all false

    // Using this constructor, Selector s("10") would have s[0] = true
    // and [1] = false;
    explicit Selector(const std::string &zeros_and_ones);
    Selector(const std::vector<bool> &);
    Selector(const std::vector<uint> &pos, uint n);
    Selector(const Selector & );
    Selector & operator=(const Selector &rhs);

    Selector & add(uint i);
    Selector & operator+=(uint i){return add(i);}
    Selector & operator+=(const Selector &rhs);   // union
    Selector & drop(uint i);
    Selector & operator-=(uint i){return drop(i);}
    Selector & operator-=(const Selector &rhs);
    Selector & operator*=(const Selector &rhs);  // intersection
    Selector & flip(uint i);

    Selector complement()const;

    void drop_all();
    void add_all();

    void swap(Selector &rhs);

    uint nvars()const;          // =="n"
    uint nvars_possible()const; // =="N"
    uint nvars_excluded()const; // == N-n;
    bool inc(uint i)const;

    bool covers(const Selector &rhs)const;
    // this->covers(rhs) iff every variable in rhs is also in *this
    bool operator==(const Selector &rhs)const;
    bool operator!=(const Selector &rhs)const;

    // Returns the set union:  locations which are in either Selector.
    Selector Union(const Selector &rhs)const;

    // Returns the set intersection, locations which are in both Selectors.
    Selector intersection(const Selector &rhs)const;

    // Returns a Selector that is 1 in places where this disagrees with rhs.
    Selector exclusive_or(const Selector &rhs)const;
    Selector & cover(const Selector &rhs);  // makes *this cover rhs

    uint indx(uint i)const;  // i=0..n-1, ans in 0..N-1
    uint INDX(uint I)const;  // I=0..N-1, ans in 0..n-1

    Vec vec()const;    // explicit vector  of 0's and 1's

    Vec select(const Vec &x)const;          // x includes intercept
    Vec select_add_int(const Vec &x)const;  // intercept is implicit
    Spd select(const Spd &)const;
    Mat select_cols(const Mat &M)const;
    Mat select_cols_add_int(const Mat &M)const;
    Mat select_square(const Mat &M)const;  // selects rows and columns
    Mat select_rows(const Mat &M)const;

    Vec select(const VectorView & x)const;
    Vec select(const ConstVectorView & x)const;

    Vec expand(const Vec &x)const;
    Vec expand(const VectorView &x)const;
    Vec expand(const ConstVectorView &x)const;

    Vec & zero_missing_elements(Vec &v)const;

    // Fill ans with select_cols(M) * select(v).
    void sparse_multiply(const Matrix &M, const Vector &v, VectorView ans)const;
    Vector sparse_multiply(const Matrix &M, const Vector &v)const;
    Vector sparse_multiply(const Matrix &M, const VectorView &v)const;
    Vector sparse_multiply(const Matrix &M, const ConstVectorView &v)const;

    template <class T>
    std::vector<T> select(const std::vector<T> &v)const;

    template<class T>
    T sub_select(const T &x, const Selector &rhs)const;
    // x is an object obtained by select(original_object).
    // this->covers(rhs).  sub_select(x,rhs) returns the object that
    // would have been obtained by rhs.select(original_object)

    // Returns the index of a randomly selected included (or excluded)
    // element.  If no (all) elements are included then -1 is returned
    // as an error code.
    int random_included_position(RNG &rng)const;
    int random_excluded_position(RNG &rng)const;
  };
  //______________________________________________________________________

  ostream & operator<<(ostream &, const Selector &);
  istream & operator>>(istream &, Selector &);

  // set difference, union, intersection
  Selector operator-(const Selector & lhs, const Selector &rhs);
  Selector operator+(const Selector & lhs, const Selector &rhs);
  Selector operator*(const Selector & lhs, const Selector &rhs);

  Vec operator*(const Selector &inc, double x);
  Vec operator*(double x, const Selector &inc);

  Selector find_contiguous_subset(const Vec &big, const Vec &small);
  // find_contiguous_subset returns the indices of 'big' that contain
  // the vector 'small.'  If 'small' is not found then an empty
  // include is returned.

  Selector append(bool, const Selector &);
  Selector append(const Selector &, bool);
  Selector append(const Selector &, const bool &);

  template <class T>
  std::vector<T> select(const std::vector<T> &v, const std::vector<bool> &vb){
    uint n = v.size();
    assert(vb.size()==n);
    std::vector<T> ans;
    for(uint i=0; i<n; ++i) if(vb[i]) ans.push_back(v[i]);
    return ans;
  }

  template <class T>
  std::vector<T> Selector::select(const std::vector<T> &v)const{
    assert(v.size()==nvars_possible());
    std::vector<T> ans;
    ans.reserve(nvars());
    for(uint i=0; i<nvars_possible(); ++i)
      if(inc(i))
	ans.push_back(v[i]);
    return ans;}

  template <class T>
  T Selector::sub_select(const T &x, const Selector &rhs)const{
    assert(rhs.nvars() <= this->nvars());
    assert(this->covers(rhs));

    Selector tmp(nvars(), false);
    for(uint i=0; i<rhs.nvars(); ++i){
      tmp.add(INDX(rhs.indx(i)));
    }
    return tmp.select(x);
  }

}  // ends namespace BOOM
#endif // BOOM_SELECTOR_HPP
