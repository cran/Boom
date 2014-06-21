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

#ifndef BOOM_HIERARCHICAL_POISSON_MODEL_HPP_
#define BOOM_HIERARCHICAL_POISSON_MODEL_HPP_

#include <Models/PoissonModel.hpp>
#include <Models/GammaModel.hpp>
#include <Models/Policies/CompositeParamPolicy.hpp>

namespace BOOM {

  class HierarchicalPoissonData : public Data {
   public:
    HierarchicalPoissonData(double event_count, double exposure);
    HierarchicalPoissonData * clone()const;
    virtual ostream & display(ostream &out)const;
    double event_count() const {return event_count_;}
    double exposure() const {return exposure_;}
   private:
    double event_count_;
    double exposure_;
  };

  class HierarchicalPoissonModel
      : public CompositeParamPolicy,
        public PriorPolicy {
   public:

    HierarchicalPoissonModel(double lambda_prior_guess,
                             double lambda_prior_sample_size);

    HierarchicalPoissonModel(Ptr<GammaModel> prior_model);

    HierarchicalPoissonModel(const HierarchicalPoissonModel &rhs);

    virtual HierarchicalPoissonModel * clone()const;

    void add_data_level_model(Ptr<PoissonModel> data_level_model);

    // Removes all data_level_models and their associated parameters
    // and data.
    virtual void clear_data();

    // Clear the data from all data_level_models, but does not delete
    // the models.
    void clear_client_data();

    // Clear the learning methods for each of the client models.
    virtual void clear_methods();

    // Adds the data_level_models from rhs to this.
    virtual void combine_data(const Model &rhs, bool just_suf = true);

    // Creates a new data_level_model with data assigned.
    virtual void add_data(Ptr<Data>);

    // Returns the number of data_level_models managed by this model.
    int number_of_groups()const;

    PoissonModel * data_model(int which_group);
    const PoissonModel * data_model(int which_group)const;
    GammaModel * prior_model();
    const GammaModel * prior_model()const;

    double prior_mean()const;
    double prior_sample_size()const;

   private:
    void initialize();
    Ptr<GammaModel> prior_;
    std::vector<Ptr<PoissonModel> > data_level_models_;
  };
}  // namespace BOOM

#endif //  BOOM_HIERARCHICAL_POISSON_MODEL_HPP_
