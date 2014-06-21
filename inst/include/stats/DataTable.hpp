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

#ifndef BOOM_DATA_TABLE_HPP
#define BOOM_DATA_TABLE_HPP

#include <BOOM.hpp>

#include <LinAlg/Vector.hpp>
#include <LinAlg/Matrix.hpp>
#include <LinAlg/Types.hpp>

#include <Models/DataTypes.hpp>
#include <Models/CategoricalData.hpp>
#include <limits>

namespace BOOM{
  class DesignMatrix;

   class DataTable{
     // A DataTable is created by reading a plain text file and
     // storing "variables" in a table.  Variables can be extracted
     // from the DataTable either individually (e.g. to get the y
     // variable for a regression/classification problem) or as a
     // DesignMatrix.  When building the design matrix, special care
     // is taken to properly name dummy variables.

   public:
     typedef CategoricalData catdat;
     typedef std::vector<Ptr<catdat> > CatVec;
     typedef std::vector<double> dvector;
     enum variable_type {unknown= -1, continuous, categorical};
     typedef std::vector<string> StringVec;

     //--- constructors ---
     DataTable(const string &fname, bool header=false,
 	       const string &sep="");

     //--- size  ---
     uint nvars()const; // number of variables stored in the table
     uint nobs()const;  // number of observations
     uint nlevels(uint i)const;  // 1 for continuous, nlevels for categorical

     //--- look inside ---
     ostream & print(ostream &out,
		     uint from =0,
		     uint to= std::numeric_limits<uint>::max())const;
     StringVec & vnames();
     const StringVec & vnames()const;

     //--- extract variables ---
     Vec getvar(uint n, uint count_from=0)const;
     // obs values for cont, uint values for cat

     std::vector<Ptr<catdat> > get_nominal(uint n, uint count_from=0)const;
     // counting from 0

     std::vector<Ptr<OrdinalData> >
     get_ordinal(uint n, uint count_from=0)const;
     std::vector<Ptr<OrdinalData> >
     get_ordinal(uint n, const StringVec &ord, uint count_from=0)const;

     //--- build a design matrix ---
     DesignMatrix design(bool add_icpt = false)const;
     DesignMatrix design(const std::vector<bool> &include,
 			 bool add_icpt = false)const;
     DesignMatrix design(std::vector<uint> include,
 			 bool add_icpt = false,
			 uint counting_from=0)const;

   private: //--------------------------------------------------
     std::vector<Vec> cont_vars;
     std::vector<CatVec> cat_vars;

     std::vector<variable_type> vtypes;
     StringVec vnames_;
     void diagnose_types(const StringVec &);
     bool check_type(variable_type, const string &s)const;
     //    uint compute_variable_index(uint n)const;
   };

  ostream & operator<<(ostream &out, const DataTable &dt);

}
#endif // BOOM_DATA_TABLE_HPP
