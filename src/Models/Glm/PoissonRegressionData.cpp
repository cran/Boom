/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#include <Models/Glm/PoissonRegressionData.hpp>
#include <cpputil/report_error.hpp>

namespace BOOM {

  PoissonRegressionData::PoissonRegressionData(int y, const Vec &x)
      : GlmData<IntData>(y, x),
        exposure_(1.0),
        log_exposure_(0)
  {}

  PoissonRegressionData::PoissonRegressionData(int y, const Vec &x, double exposure)
      : GlmData<IntData>(y, x),
        exposure_(exposure),
        log_exposure_(log(exposure))
  {
    if (exposure < 0) {
      report_error("You can't pass a negative exposure to the "
                   "PoissonRegressionData constructor.");
    }
  }

  PoissonRegressionData * PoissonRegressionData::clone()const{
    return new PoissonRegressionData(*this);}

  ostream & PoissonRegressionData::display(ostream &out)const{
    out << "[" << exposure_ << "]  ";
    return GlmData<IntData>::display(out);
  }

  double PoissonRegressionData::exposure()const{return exposure_;}

  double PoissonRegressionData::log_exposure()const{return log_exposure_;}

}
