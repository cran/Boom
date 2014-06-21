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

#ifndef BOOM_COMPOSITE_EM_MIXTURE_COMPONENT_HPP
#define BOOM_COMPOSITE_EM_MIXTURE_COMPONENT_HPP

#include <Models/CompositeModel.hpp>
#include <Models/EmMixtureComponent.hpp>

namespace BOOM{

  // A composite model class that can be used in a finite mixture that
  // expects to be fit using the EM algorithm.  The primary methods
  // needed here are find_posterior_mode() and add_mixture_data(),
  // both of which operate by delegating to the composite elements.
  class CompositeEmMixtureComponent
    : public CompositeModel,
      public EmMixtureComponent
  {
  public:
    typedef EmMixtureComponent EM;

    CompositeEmMixtureComponent();

    template <class MOD>
    CompositeEmMixtureComponent(const std::vector<Ptr<MOD> > &mod)
        : CompositeModel(mod),
          em_components_(mod.begin(), mod.end())
    {}

    CompositeEmMixtureComponent(const CompositeEmMixtureComponent &rhs);

    virtual CompositeEmMixtureComponent * clone()const;
    virtual void mle();
    virtual void find_posterior_mode();
    virtual void add_mixture_data(Ptr<Data>, double prob);

    // add_model will report an error if called with a
    // MixtureComponent that is not also an EmMixtureComponent.
    virtual void add_model(Ptr<MixtureComponent>);

    // Define members that are ambiguous because of multiple
    // inheritance.
    virtual double pdf(Ptr<Data> dp, bool logscale)const{
      return CompositeModel::pdf(dp, logscale);}
    virtual double pdf(const Data * dp, bool logscale)const{
      return CompositeModel::pdf(dp, logscale);}
  private:
    // em_components_ points to the same thing as m_ in the
    // CompositeModel base class.
    std::vector<Ptr<EmMixtureComponent> > em_components_;
  };

}
#endif// BOOM_COMPOSITE_EM_MIXTURE_COMPONENT_HPP
