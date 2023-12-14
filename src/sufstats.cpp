/*
  Copyright (C) 2005-2017 Steven L. Scott

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

#include "r_interface/sufstats.hpp"
#include "r_interface/boom_r_tools.hpp"
#include "Models/Glm/RegressionModel.hpp"

namespace BOOM {
  namespace RInterface {
    RegSuf * CreateRegSuf(SEXP r_reg_suf) {
      return new NeRegSuf(ToBoomSpdMatrix(getListElement(r_reg_suf, "xtx")),
                          ToBoomVector(getListElement(r_reg_suf, "xty")),
                          Rf_asReal(getListElement(r_reg_suf, "yty")),
                          Rf_asReal(getListElement(r_reg_suf, "n")),
                          Rf_asReal(getListElement(r_reg_suf, "ybar")),
                          ToBoomVector(getListElement(r_reg_suf, "xbar")));
    }
  }  // namespace RInterface
}  // namespace BOOM
