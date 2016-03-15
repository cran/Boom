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
#include <cpputil/Split.hpp>
#include <cpputil/string_utils.hpp>
#include <cpputil/report_error.hpp>
#include <cctype>
#include <string>
#include <vector>
#include <boost/tokenizer.hpp>

namespace BOOM{
  using std::vector;
  using std::string;

  StringSplitter::StringSplitter(string s, bool allow_quotes)
    : delim(s),
      quotes(allow_quotes? "\"'" : ""),
      delimited(!is_all_white(s))
  {
  }

  //------------------------------------------------------------
  vector<string> StringSplitter::operator()(const string &s)const{

    typedef boost::escaped_list_separator<char> Sep;
    typedef boost::tokenizer<Sep> tokenizer;

    try{
      Sep sep("", delim, quotes);
      tokenizer tk(s, sep);
      if(delimited){
        vector<string> ans(tk.begin(), tk.end());
        return ans;
      }

      vector<string> ans;
      for(tokenizer::iterator it = tk.begin(); it!=tk.end(); ++it){
        string s = *it;
        if(s.size()>0) ans.push_back(s);
      }
      return ans;

    }catch (std::exception &e){
      report_error(e.what());
    }catch(...){
      report_error(
          "caught unknown exception in StringSplitter::operator()");
    }
    vector<string> result;  // never get here
    return result;
  }

}
