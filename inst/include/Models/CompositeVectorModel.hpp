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

#include <Models/CompositeData.hpp>
#include <Models/DoubleModel.hpp>
#include <Models/Policies/CompositeParamPolicy.hpp>
#include <Models/Policies/IID_DataPolicy.hpp>
#include <Models/Policies/PriorPolicy.hpp>
#include <Models/VectorModel.hpp>

namespace BOOM {

  // A class for forming a VectorModel out of a bunch of DoubleModel's.
  class CompositeVectorModel
      : public VectorModel,
        public CompositeParamPolicy,
        public IID_DataPolicy<CompositeData>,
        public PriorPolicy
  {
   public:
    CompositeVectorModel();
    CompositeVectorModel(const CompositeVectorModel &rhs);
    CompositeVectorModel(CompositeVectorModel &&rhs) = default;
    CompositeVectorModel & operator=(const CompositeVectorModel &rhs);

    CompositeVectorModel * clone() const override;
    double logp(const Vector &x) const override;
    Vector sim() const override;

    void add_model(Ptr<DoubleModel> model);
    void add_data(Ptr<Data> dp) override;
    void add_data(Ptr<CompositeData> dp) override;
    void clear_data() override;

   private:
    std::vector<Ptr<DoubleModel>> component_models_;
  };

}  // namespace BOOM
