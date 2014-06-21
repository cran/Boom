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
#include <algorithm>
#include <numeric>

#include <TargetFun/LoglikeSubset.hpp>
#include <Models/ModelTypes.hpp>
#include <cpputil/ParamHolder.hpp>


namespace BOOM{

  LoglikeSubsetTF::LoglikeSubsetTF(LoglikeModel *m)
    : mod(m),
      t(m->t())
  {}
  LoglikeSubsetTF::LoglikeSubsetTF(LoglikeModel *m, const ParamVec & v)
    : mod(m),
      t(v)
  {}
  LoglikeSubsetTF::LoglikeSubsetTF(LoglikeModel *m, Ptr<Params> T)
    : mod(m),
      t(1,T){}

  ParamVecHolder LoglikeSubsetTF::hold_params(const Vec &x)const{
    ParamVecHolder ans(x, t, wsp);
    return ans;
  }

  double LoglikeSubsetTF::operator()(const Vec &x)const{
    ParamVecHolder ph(hold_params(x));
    double ans = mod->loglike();
    return ans;
  }

  //------------------------------------------------------------
  dLoglikeSubsetTF::dLoglikeSubsetTF(dLoglikeModel *d)
    : LoglikeSubsetTF(d),
      dmod(d)
  {
    get_pos();
  }

  dLoglikeSubsetTF::dLoglikeSubsetTF(dLoglikeModel *d, const ParamVec & T)
    : LoglikeSubsetTF(d,T),
      dmod(d)
  {
    get_pos();
  }

  dLoglikeSubsetTF::dLoglikeSubsetTF(dLoglikeModel *d, Ptr<Params> T)
    : LoglikeSubsetTF(d,T),
      dmod(d)
  {
    get_pos();
  }



  void dLoglikeSubsetTF::get_pos(){
    ParamVec v(dmod->t());
    uint Np = v.size();

    std::vector<uint> sizes(Np);
    for(uint i=0; i<Np; ++i) sizes[i] = v[i]->size();
    std::partial_sum(sizes.begin(), sizes.end(), sizes.begin());
    // sizes contains a running total of the size of the vectorized
    // params for dmod

    uint answer_size = sizes.back();
    std::vector<bool> inc(answer_size, false);

    for(uint i=0; i<Np; ++i){
      // if v[i] is in t then set the appropriate
      // elements in pos_ to 'true'
      ParamVec::iterator it =
	std::find(t.begin(), t.end(), v[i]);
      if(it != t.end()){
	uint ind = (i==0 ? 0 : sizes[i-1]);
	std::vector<bool>::iterator newit= inc.begin();
	newit+= ind;
	std::fill_n(newit, v[i]->size(), true);}}
    pos_.reset(new Selector(inc));
  }

  const Selector & dLoglikeSubsetTF::pos()const{ return *pos_;}


  double dLoglikeSubsetTF::operator()(const Vec &x, Vec &g)const{
    ParamVecHolder ph= hold_params(x);
    const Selector & inc(pos());
    Vec G(inc.nvars_possible());
    double ans = dmod->dloglike(G);
    g = inc.select(G);
    return ans;
  }

  //------------------------------------------------------------
  d2LoglikeSubsetTF::d2LoglikeSubsetTF(d2LoglikeModel *d2)
    : dLoglikeSubsetTF(d2),
      d2mod(d2)
  {}

  d2LoglikeSubsetTF::d2LoglikeSubsetTF(d2LoglikeModel *d2,
			   Ptr<Params> T)
    : dLoglikeSubsetTF(d2,T),
      d2mod(d2)
  {}

  d2LoglikeSubsetTF::d2LoglikeSubsetTF(d2LoglikeModel *d2, const ParamVec &T)
    : dLoglikeSubsetTF(d2,T),
      d2mod(d2)
  {}

  double d2LoglikeSubsetTF::operator()(const Vec &x, Vec &g, Mat &h)const{
    ParamVecHolder ph = hold_params(x);
    const Selector & inc(pos());
    uint N = inc.nvars_possible();
    Vec G(N);
    Mat H(N,N);
    double ans = d2mod->d2loglike(G, H);
    g = inc.select(G);
    h = inc.select_square(H);
    return ans;
  }

}
