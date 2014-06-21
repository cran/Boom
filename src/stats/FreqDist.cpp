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
#include <stats/FreqDist.hpp>
#include <Models/CategoricalData.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cpputil/report_error.hpp>

namespace BOOM{

  namespace {

    template <class INT> std::string u2str(INT u){
      std::ostringstream out;
      out << u;
      return out.str();
    }

    template <class INT> std::vector<int>
    count_values(const std::vector<INT> &y, std::vector<std::string> &labels) {
      std::vector<int> x(y.begin(), y.end());
      std::vector<int> counts;
      labels.clear();
      std::sort(x.begin(), x.end());
      int last = x[0];
      string lab = u2str(last);
      uint count = 1;
      for(uint i=1; i<x.size(); ++i){
        if(x[i]!=last){
	counts.push_back(count);
	labels.push_back(lab);
	count = 1;
	last = x[i];
	lab = u2str(last);
      }else{
	++count;
      }
    }
    counts.push_back(count);
    labels.push_back(lab);
    return counts;
    }
  }

  FreqDist::FreqDist(const std::vector<uint> &y){
    counts_ = count_values(y, labs_);
  }

  FreqDist::FreqDist(const std::vector<int> &y){
    counts_ = count_values(y, labs_);
  }

  FreqDist::FreqDist(const std::vector<long> &y){
    counts_ = count_values(y, labs_);
  }

  FreqDist::FreqDist(const std::vector<unsigned long> &y){
    counts_ = count_values(y, labs_);
  }

  std::ostream & FreqDist::print(std::ostream &out)const{
    uint N = labs_.size();
    uint labfw=0;
    uint countfw=0;
    for(uint i = 0; i<N; ++i){
      uint len = labs_[i].size();
      if(len > labfw) labfw = len;

      string s = u2str(counts_[i]);
      len = s.size();
      if(len > countfw) countfw = len;
    }
    labfw += 2;
    countfw +=2;

    for(uint i=0; i<N; ++i){
      out << std::setw(labfw) << labs_[i]
	  << std::setw(countfw) << counts_[i]
	  << std::endl;
    }
    return out;
  }

  void FreqDist::reset(const std::vector<int> &counts,
                 const std::vector<std::string> &labels){
    if(counts.size() != labels.size()){
      report_error("counts and labels must be the same size in FreqDist::reset");
    }
    counts_ = counts;
    labs_ = labels;
  }
}
