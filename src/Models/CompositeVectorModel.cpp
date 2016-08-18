/*
  Copyright (C) 2005-2016 Steven L. Scott

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

#include <Models/CompositeVectorModel.hpp>
#include <cpputil/report_error.hpp>

namespace BOOM {

  CompositeVectorModel::CompositeVectorModel() {}

  CompositeVectorModel::CompositeVectorModel(const CompositeVectorModel &rhs) {
    for (int i = 0; i < rhs.component_models_.size(); ++i) {
      add_model(rhs.component_models_[i]->clone());
    }
  }

  CompositeVectorModel &
  CompositeVectorModel::operator=(const CompositeVectorModel &rhs) {
    if (&rhs != this) {
      component_models_.clear();
      ParamPolicy::clear();
      DataPolicy::clear_data();
      PriorPolicy::clear_methods();
      for (int i = 0; i < rhs.component_models_.size(); ++i) {
        add_model(rhs.component_models_[i]->clone());
      }
    }
    return *this;
  }

  CompositeVectorModel * CompositeVectorModel::clone() const {
    return new CompositeVectorModel(*this);
  }

  double CompositeVectorModel::logp(const Vector &x) const {
    if (x.size() != component_models_.size()) {
      report_error("Wrong size argument passed to CompositeVectorModel::logp.");
    }
    double ans = 0;
    for (int i = 0; i < component_models_.size(); ++i) {
      ans += component_models_[i]->logp(x[i]);
    }
    return ans;
  }

  Vector CompositeVectorModel::sim() const {
    Vector ans(component_models_.size());
    for (int i = 0; i < component_models_.size(); ++i) {
      ans[i] = component_models_[i]->sim();
    }
    return ans;
  }

  void CompositeVectorModel::add_model(Ptr<DoubleModel> m) {
    component_models_.push_back(m);
    ParamPolicy::add_model(m);
  }

  void CompositeVectorModel::add_data(Ptr<Data> dp) {
    return add_data(dp.dcast<CompositeData>());
  }

  void CompositeVectorModel::add_data(Ptr<CompositeData> dp) {
    if (dp->dim() != component_models_.size()) {
      report_error("Wrong size data passed to CompositeVectorModel::add_data.");
    }
    for (int i = 0; i < dp->dim(); ++i) {
      component_models_[i]->add_data(dp->get_ptr(i));
    }
    DataPolicy::add_data(dp);
  }

  void CompositeVectorModel::clear_data() {
    for (int i = 0; i < component_models_.size(); ++i) {
      component_models_[i]->clear_data();
    }
    DataPolicy::clear_data();
  }

}  // namespace BOOM
