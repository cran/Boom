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
#ifndef BOOM_FREQ_DIST_HPP
#define BOOM_FREQ_DIST_HPP

#include <vector>
#include <string>

namespace BOOM{

  class FreqDist{
   public:
    FreqDist(const std::vector<unsigned int> &y);
    FreqDist(const std::vector<int> &y);
    FreqDist(const std::vector<long> &y);
    FreqDist(const std::vector<unsigned long> &y);

    const std::vector<std::string> & labels()const{
      return labs_;}
    const std::vector<int> & counts()const{
      return counts_; }

    std::ostream & print(std::ostream &out)const;
   private:
    std::vector<std::string> labs_;
    std::vector<int> counts_;
   protected:
    FreqDist(){}
    void reset(const std::vector<int> &counts,
               const std::vector<std::string> &labels);
  };

  inline std::ostream & operator<<(std::ostream &out, const FreqDist &f){
    return f.print(out);
  }
}
#endif// BOOM_FREQ_DIST_HPP
