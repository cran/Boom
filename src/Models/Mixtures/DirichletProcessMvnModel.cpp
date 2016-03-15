/*
  Copyright (C) 2005-2015 Steven L. Scott

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

#include <Models/Mixtures/DirichletProcessMvnModel.hpp>
#include <distributions.hpp>
#include <cpputil/report_error.hpp>

namespace BOOM {

  namespace {
    typedef DirichletProcessMvnModel DPMM;
  }  // namespace

  DPMM::DirichletProcessMvnModel(int dim, double alpha)
      : alpha_(new UnivParams(alpha)),
        dim_(dim)
  {
    if (dim <= 0) {
      report_error("Dimension must be at least one for "
                   "DirichletProcessMvnModel.");
    }
    register_models();
  }

  DPMM::DirichletProcessMvnModel(const DPMM &rhs)
      : alpha_(rhs.alpha_->clone()),
        mixture_components_(rhs.mixture_components_),
        dim_(rhs.dim_)
  {
    for (int i = 0; i < mixture_components_.size(); ++i) {
      mixture_components_[i] = rhs.mixture_components_[i]->clone();
    }
    register_models();
  }

  DPMM * DPMM::clone() const {return new DPMM(*this);}

  int DPMM::dim() const {
    return dim_;
  }

  int DPMM::number_of_clusters() const {
    return mixture_components_.size();
  }

  double DPMM::alpha() const {
    return alpha_->value();
  }

  void DPMM::set_alpha(double alpha) {
    alpha_->set(alpha);
  }

  void DPMM::assign_data_to_cluster(const Vector &y, int cluster) {
    if (cluster < mixture_components_.size()) {
      mixture_components_[cluster]->suf()->update_raw(y);
    } else if (cluster == mixture_components_.size()) {
      NEW(MvnModel, new_cluster)(dim_);
      new_cluster->suf()->update_raw(y);
      mixture_components_.push_back(new_cluster);
      ParamPolicy::add_model(new_cluster);
    } else {
      report_error("Cluster indicator out of range in "
                   "assign_data_to_cluster.");
    }
  }

  void DPMM::remove_data_from_cluster(const Vector &y, int cluster) {
    if (cluster < mixture_components_.size()) {
      Ptr<MvnModel> mvn = mixture_components_[cluster];
      mvn->suf()->remove_data(y);
      if (mvn->suf()->n() == 0) {
        ParamPolicy::drop_model(mvn);
        mixture_components_.erase(mixture_components_.begin() + cluster);
      }
    } else {
      report_error("Cluster indicator out of range in "
                   "remove_data_from_cluster.");
    }
  }

  const MvnModel & DPMM::cluster(int i) const {
    if (i >= mixture_components_.size()) {
      report_error("Cluster indicator out of range in cluster().");
    }
    return *mixture_components_[i];
  }

  void DPMM::set_component_params(int cluster,
                                  const Vector &mu,
                                  const SpdMatrix &Siginv) {
    Ptr<MvnModel> mvn = mixture_components_[cluster];
    mvn->set_mu(mu);
    mvn->set_siginv(Siginv);
  }

  void DPMM::register_models() {
    ParamPolicy::clear();
    ParamPolicy::add_params(alpha_);
    for (int i = 0; i < mixture_components_.size(); ++i) {
      ParamPolicy::add_model(mixture_components_[i]);
    }
  }

}  // namespace BOOM
