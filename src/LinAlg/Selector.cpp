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

#include <LinAlg/Selector.hpp>
#include <LinAlg/Vector.hpp>
#include <LinAlg/Matrix.hpp>
#include <LinAlg/SpdMatrix.hpp>
#include <LinAlg/Types.hpp>
#include <cpputil/seq.hpp>

#include <distributions.hpp>

#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace BOOM {

  namespace {
    typedef std::vector<bool> vb;
    typedef std::vector<uint> vpos;

    vb s2vb(const std::string &);

    vb s2vb(const std::string & s){
      uint n = s.size();
      std::vector<bool> ans(n,false);
      for(uint i=0; i<n; ++i){
        char c = s[i];
        if(c=='1') ans[i] = true;
        else if(c=='0') ans[i] = false;
        else{
          ostringstream err;
          err << "only 0's and 1's are allowed in the 'Selector' string constructor "
              << endl
              << "you supplied:  "  << endl
              << s << endl
              << "first illegal value found at position " << i << "." << endl;
          throw_exception<std::runtime_error>(err.str());
        }
      }
      return ans;
    }
  }

  void Selector::reset_inc_indx(){
    inc_indx.clear();
    for(uint i=0; i<nvars_possible(); ++i)
      if(inc(i)) inc_indx.push_back(i);
  }

  Selector::Selector(){}

  Selector::Selector(uint p, bool all)
      : std::vector<bool>(p,all),
        include_all(all)
  {
    reset_inc_indx();
  }

  Selector::Selector(const std::string &s)
      : std::vector<bool>(s2vb(s)),
        include_all(false)
  {
    if(nvars()==nvars_possible()) include_all=true;
    reset_inc_indx();
  }

  Selector::Selector(const vb& in)
      : std::vector<bool>(in),
        include_all(false)
  {
    reset_inc_indx();
  }

  Selector::Selector(const std::vector<uint> &pos, uint n)
      : std::vector<bool>(n, false),
        inc_indx(),
        include_all(false)
  {
    for(uint i=0; i<pos.size(); ++i) add(pos[i]);
  }

  Selector::Selector(const Selector &rhs)
      : std::vector<bool>(rhs),
        inc_indx(rhs.inc_indx),
        include_all(rhs.include_all)
  {}

  Selector & Selector::operator=(const Selector &rhs){
    if(&rhs==this) return *this;
    std::vector<bool>::operator=(rhs);
    inc_indx = rhs.inc_indx;
    include_all =rhs.include_all;
    return *this;
  }

  void Selector::check_size_eq(uint p, const string &fun)const{
    if(p==nvars_possible()) return;
    ostringstream err;

    err << "error in function Selector::" << fun << endl
	<< "Selector::nvars_possible()== " << nvars_possible() << endl
	<< "you've assumed it to be " << p << endl;
    throw_exception<std::runtime_error>(err.str());
  }

  void Selector::check_size_gt(uint p, const string &fun)const{
    if(p< nvars_possible()) return;
    ostringstream err;

    err << "error in function Selector::" << fun << endl
	<< "Selector::nvars_possible()== " << nvars_possible() << endl
	<< "you tried to access element " << p << endl;
    throw_exception<std::runtime_error>(err.str());
  }

  Selector & Selector::add(uint p){
    check_size_gt(p, "add");
    if(include_all) return *this;
    if(inc(p)==false){
      (*this)[p] = true;
      vpos::iterator it =
 	std::lower_bound(inc_indx.begin(), inc_indx.end(), p);
      inc_indx.insert(it, p);}
    return *this;
  }

  void Selector::drop_all(){
    include_all = false;
    inc_indx.clear();
    std::vector<bool>::assign(size(), false);
  }

  void Selector::add_all(){
    include_all=true;
    uint n = nvars_possible();
    inc_indx = seq<uint>(0, n-1);
    std::vector<bool>::assign(n, true);
  }

  Selector &Selector::drop(uint p){
    check_size_gt(p, "drop");
    if(include_all){
      reset_inc_indx();
      include_all=false;
    }
    if(inc(p)){
      (*this)[p] = false;
      vpos::iterator it =
 	std::lower_bound(inc_indx.begin(), inc_indx.end(), p);
      inc_indx.erase(it);}
    return *this;
  }

  Selector & Selector::flip(uint p){
    if(inc(p)) drop(p);
    else add(p);
    return *this;
  }


  Selector Selector::complement()const{
    Selector ans(*this);
    for(uint i=0; i<nvars_possible(); ++i){
      ans.flip(i);
    }
    return ans;
  }

  void Selector::swap(Selector &rhs){
    std::vector<bool>::swap(rhs);
    std::swap(inc_indx, rhs.inc_indx);
    std::swap(include_all, rhs.include_all);
  }

  bool Selector::inc(uint i)const{ return (*this)[i];}

  uint Selector::nvars()const{
    return include_all ? nvars_possible() : inc_indx.size(); }

  uint Selector::nvars_possible()const{return size();}

  uint Selector::nvars_excluded()const{return nvars_possible() - nvars();}

  uint Selector::indx(uint i)const{
    if(include_all) return i;
    return inc_indx[i]; }
  uint Selector::INDX(uint i)const{
    if(include_all) return i;
    std::vector<uint>::const_iterator loc =
      std::lower_bound(inc_indx.begin(), inc_indx.end(), i);
    return loc - inc_indx.begin();
  }

  Vec Selector::vec()const{
    Vec ans(nvars_possible(), 0.0);
    uint n = nvars();
    for(uint i=0; i<n; ++i){
      uint I = indx(i);
      ans[I] = 1;
    }
    return ans;
  }

  bool Selector::covers(const Selector &rhs)const{
    for(uint i=0; i<rhs.nvars(); ++i){
      uint I = rhs.indx(i);
      if(!inc(I)) return false;}
    return true;}

  bool Selector::operator==(const Selector &rhs)const{
    const std::vector<bool> & RHS(rhs);
    const std::vector<bool> & LHS(*this);
    return LHS == RHS;
  }
  bool Selector::operator!=(const Selector &rhs)const{
    return ! operator==(rhs);}

  Selector Selector::Union(const Selector &rhs)const{
    uint n = nvars_possible();
    check_size_eq(rhs.nvars_possible(), "Union");
    Selector ans(n, false);
    for(uint i=0; i<n; ++i) if(inc(i) || rhs.inc(i)) ans.add(i);
    return ans;
  }

  Selector Selector::intersection(const Selector &rhs)const{
    uint n = nvars_possible();
    check_size_eq(rhs.nvars_possible(), "intersection");
    Selector ans(n, false);
    const Selector &shorter(rhs.nvars() < nvars()? rhs: *this);
    const Selector &longer(rhs.nvars()<nvars() ? *this : rhs);

    for(uint i=0; i<shorter.nvars(); ++i){
      uint I = shorter.indx(i);     // I is included in shorter
      if(longer.inc(I))             // I is included in longer
	ans.add(I);
    }
    return ans;}


  // Returns a Selector of the same size as this, which is 1 where this[i] != that[i]
  Selector Selector::exclusive_or(const Selector &that)const{
    uint n = nvars_possible();
    check_size_eq(that.nvars_possible(), "intersection");
    Selector ans(n, false);
    for(int i = 0; i < n; ++i){
      ans[i] = (*this)[i] != that[i];
    }
    return ans;
  }

  Selector & Selector::cover(const Selector &rhs){
    check_size_eq(rhs.nvars_possible(), "cover");
    for(uint i=0; i<rhs.nvars(); ++i)
      add(rhs.indx(i));  // does nothing if already added
    return *this;
  }

  template <class V>
  Vec inc_select(const V &x, const Selector &inc){
    uint nx = x.size();
    uint N = inc.nvars_possible();
    if(nx != N){
      ostringstream msg;
      msg << "Selector::select... x.size() = " << nx << " nvars_possible() = "
	  << N << endl;
      throw_exception<std::runtime_error>(msg.str());
    }
    uint n = inc.nvars();

    if(n==N) return x;
    Vec ans(n);
    for(uint i=0; i<n; ++i) ans[i] = x[inc.indx(i)];
    return ans;
  }

  Vec Selector::select(const Vec &x)const{
    return inc_select<Vec>(x, *this); }
  Vec Selector::select(const VectorView &x)const{
    return inc_select<VectorView>(x, *this); }
  Vec Selector::select(const ConstVectorView &x)const{
    return inc_select<ConstVectorView>(x, *this); }

  template <class V>
  Vec inc_expand(const V &x, const Selector &inc){
    uint n = inc.nvars();
    uint nx = x.size();
    if(nx!=n){
      ostringstream msg;
      msg << "Selector::expand... x.size() = " << nx << " nvars() = "
	  << n << endl;
      throw_exception<std::runtime_error>(msg.str());
    }
    uint N = inc.nvars_possible();
    if(n==N) return x;
    Vec ans(N, 0);
    for(uint i=0; i<n; ++i){
      uint I = inc.indx(i);
      ans[I] = x[i];
    }
    return ans;
  }

  Vec Selector::expand(const Vec & x)const{
    return inc_expand(x,*this); }
  Vec Selector::expand(const VectorView & x)const{
    return inc_expand(x,*this); }
  Vec Selector::expand(const ConstVectorView & x)const{
    return inc_expand(x,*this); }


  Vec Selector::select_add_int(const Vec &x)const{
    assert(x.size()==nvars_possible()-1);
    if(include_all) return concat(1.0,x);
    Vec ans(nvars());
    ans[0]= inc(0) ? 1.0 : x[indx(0)-1];
    for(uint i=1; i<nvars(); ++i) ans[i] = x[indx(i)-1];
    //    for(uint i=1; i<nvars(); ++i) ans[i] = x[indx(i)];
    //    if(need_lb) const_cast<Vec &>(x).set_lower_bound(lb);
    return ans;
  }
  //----------------------------------------------------------------------
  Spd Selector::select(const Spd &S)const{
    uint n = nvars();
    uint N = nvars_possible();
    check_size_eq(S.ncol(), "select");
    if(include_all || n==N) return S;
    Spd ans(n);
    for(uint i=0; i<n; ++i){
      uint I = inc_indx[i];
      const double * s(S.col(I).data());
      double * a(ans.col(i).data());
      for(uint j=0; j<n; ++j) a[j] = s[inc_indx[j]];
    }
    return ans;
  }
  //----------------------------------------------------------------------
  Mat Selector::select_cols(const Mat &m)const{
    if(include_all) return m;
    Mat ans(m.nrow(), nvars());
    for(uint i=0; i<nvars(); ++i){
      uint I=indx(i);
      std::copy(m.col_begin(I), m.col_end(I), ans.col_begin(i));
    }
    return ans;
  }

  Mat Selector::select_rows(const Mat &m)const{
    if(include_all) return m;
    uint n = nvars();
    Mat ans(n, m.ncol());
    for(uint i=0; i<n; ++i) ans.row(i) = m.row(indx(i));
    return ans;
  }

  Mat Selector::select_square(const Mat &m)const{
    assert(m.is_square());
    check_size_eq(m.nrow(), "select_square");
    if(include_all) return m;

    Mat ans(nvars(), nvars());
    for(uint i=0; i<nvars(); ++i){
      uint I = indx(i);
      for(uint j=0; j<nvars(); ++j){
 	uint J = indx(j);
 	ans(i,j) = m(I,J); }}
    return ans;
  }

  Vec & Selector::zero_missing_elements(Vec &v)const{
    uint N = nvars_possible();
    check_size_eq(v.size(), "zero_missing_elements");
    const Selector & inc(*this);
    for(uint i=0; i<N; ++i){
      if(!inc[i]) v[i] = 0;
    }
    return v;
  }

  void Selector::sparse_multiply(
      const Matrix &m, const Vector &v, VectorView ans)const{
    bool m_already_sparse = ncol(m) == nvars();
    if (!m_already_sparse) {
      check_size_eq(m.ncol(), "sparse_multiply");
    }
    bool v_already_sparse = v.size() == nvars();
    if (!v_already_sparse) {
      check_size_eq(v.size(), "sparse_multiply");
    }
    ans = 0;

    for (int i = 0; i < inc_indx.size(); ++i) {
      uint I = inc_indx[i];
      ans.axpy(m.col(m_already_sparse ? i : I),
               v[v_already_sparse ? i : I]);
    }
  }

  Vector Selector::sparse_multiply(
      const Matrix &m, const Vector &v)const{
    Vector ans(m.nrow(), 0.0);
    this->sparse_multiply(m, v, VectorView(ans));
    return ans;
  }

  Vector Selector::sparse_multiply(
      const Matrix &m, const VectorView &v)const{
    Vector ans(m.nrow(), 0.0);
    this->sparse_multiply(m, v, VectorView(ans));
    return ans;
  }

  Vector Selector::sparse_multiply(
      const Matrix &m, const ConstVectorView &v)const{
    Vector ans(m.nrow(), 0.0);
    this->sparse_multiply(m, v, VectorView(ans));
    return ans;
  }

  int Selector::random_included_position(RNG &rng)const{
    int number_included = nvars();
    if (number_included == 0) {
      return -1;
    }
    int j = random_int_mt(rng, 0, number_included - 1);
    return indx(j);
  }

  int Selector::random_excluded_position(RNG &rng)const{
    int N = nvars_possible();
    int n = nvars();
    int number_excluded = N - n;
    if(number_excluded == 0) return -1;
    if ((static_cast<double>(number_excluded) / N) >= .5) {
      // If the number of excluded variables is a large fraction of
      // the total then perform the random selection by rejection sampling.
      while (true) {
 	int j = random_int_mt(rng, 1, N-1);
 	if (!inc(j)) return j;
      }
    } else {
      int which_excluded_variable = random_int_mt(rng, 1, number_excluded);
      int number_excluded_so_far = 0;
      for (int i = 0; i < N; ++i) {
 	if (!inc(i)) {
          ++number_excluded_so_far;
 	  if (number_excluded_so_far == which_excluded_variable) {
            return i;
          }
        }
      }
    }
    return -1;
  }

  Selector &Selector::operator+=(const Selector &rhs){
    return cover(rhs);}
  Selector &Selector::operator-=(const Selector &rhs){
    check_size_eq(rhs.nvars_possible(), "operator-=");
    for(uint i=0; i<rhs.nvars(); ++i) drop(rhs.indx(i));
    return *this;}

  Selector & Selector::operator*=(const Selector &rhs){
    Selector tmp = intersection(rhs);
    this->swap(tmp);
    return *this;
  }


  ostream & operator<<(ostream &out, const Selector &inc){
    for(uint i=0; i<inc.nvars_possible(); ++i) out << inc.inc(i);
    return out;
  }

  istream & operator>>(istream &in, Selector &inc){
    string s;
    in >> s;
    uint n = s.size();
    std::vector<bool> tmp(n);
    for(uint i=0; i<n; ++i){
      if(s[i]=='0') tmp[i]=false;
      else if(s[i]=='1') tmp[i]=true;
      else throw_exception<std::runtime_error>(s+"is an illegal input value for 'Selector'");
    }
    Selector blah(tmp);
    inc.swap(blah);
    return in;
  }

  Selector operator-(const Selector &lhs, const Selector &rhs){
    assert(lhs.nvars_possible()==rhs.nvars_possible());
    Selector ans(lhs);
    for(uint i=0; i<rhs.nvars(); ++i) ans.drop(rhs.indx(i));
    return ans;
  }

  Selector operator+(const Selector &lhs, const Selector &rhs){
    return lhs.Union(rhs); }

  Selector operator*(const Selector &lhs, const Selector &rhs){
    return lhs.intersection(rhs);}

  //============================================================

  Vec operator*(double x, const Selector &inc){
    Vec ans(inc.nvars_possible(), 0.0);
    uint n = inc.nvars();
    for(uint i=0; i<n; ++i){
      uint I = inc.indx(i);
      ans[I] = x;
    }
    return ans;
  }

  Vec operator*(const Selector &inc, double x){ return x*inc; }


  //============================================================


  inline bool check_vec(const Vec &big, int pos, const Vec &small){
    for(uint i=0; i<small.size(); ++i){
      uint I = i;
      if(I >= big.size()) return false;
      if(big[pos+I]!=small[i]) return false;
    }
    return true;
  }

  Selector find_contiguous_subset(const Vec &big, const Vec &small){
    std::vector<bool> vec(big.size(), false);
    Vec::const_iterator b = big.begin();
    Vec::const_iterator it = big.begin();
    Vec::const_iterator end = big.end();

    for(uint i=0; i<small.size(); ++i){
      it=std::find(it,end, small[i]);
      uint I = it-b;
      vec[I]=true;
    }
    return Selector(vec);}


  //============================================================
  // returns a new Selector with newinc as its first element, and Inc
  // as trailing elements.  New size = Inc.size() + 1
  Selector append(bool newinc, const Selector &Inc){
    std::vector<bool>  res(Inc.nvars_possible()+1);
    const std::vector<bool>& old(Inc);
    std::vector<bool>::iterator it = res.begin();
    *it = newinc;
    ++it;
    std::copy(old.begin(), old.end(), it);
    return Selector(res);
  }

  // returns a new Selector with newinc as its last element, and Inc
  // as leading elements.  New size = Inc.size() + 1
  Selector append(const Selector &Inc, bool newinc){
    typedef std::vector<bool> vb;
    vb res(Inc.nvars_possible()+1);
    const vb& old(Inc);
    std::copy(old.begin(), old.end(), res.begin());
    res.back()=newinc;
    return Selector(res);
  }

  Selector append(const Selector &Inc1, const Selector &Inc2){
    typedef std::vector<bool> vb;
    const vb & first(Inc1);
    const vb & second(Inc2);
    vb res(first.size()+second.size());
    vb::iterator resit = copy(first.begin(), first.end(), res.begin());
    copy(second.begin(), second.end(), resit);
    return Selector(res);
  }

}
