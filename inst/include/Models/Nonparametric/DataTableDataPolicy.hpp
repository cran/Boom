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

#ifndef BOOM_DATA_TABLE_DATA_POLICY_HPP_
#define BOOM_DATA_TABLE_DATA_POLICY_HPP_

#include <Models/ModelTypes.hpp>
#include <Models/DataTypes.hpp>
#include <stats/DataTable.hpp>

namespace BOOM {

  class DataTableDataPolicy
      : virtual public Model {
   public:
    DataTableDataPolicy() {}

    DataTableDataPolicy(const DataTable &table);

    // To be like other models, copying the model does not copy the
    // data, so a copied DataTableDataPolicy starts off with a null
    // data pointer.
    DataTableDataPolicy(const DataTableDataPolicy &rhs);

    DataTableDataPolicy * clone() const override = 0;

    void add_data(Ptr<Data> dp) override;
    void clear_data() override;
    void combine_data(const Model &other_model, bool just_suf = true) override;

    const DataTable & data_table() const;
    void add_data_table(Ptr<DataTable> data);

   private:
    Ptr<DataTable> data_;
    DataTable empty_data_table_;
  };

}  // namespace BOOM

#endif //  BOOM_DATA_TABLE_DATA_POLICY_HPP_
