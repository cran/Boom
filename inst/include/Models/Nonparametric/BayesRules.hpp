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

#ifndef BOOM_NONPARAMETRIC_BAYES_RULES_HPP_
#define BOOM_NONPARAMETRIC_BAYES_RULES_HPP_

#include <Bmath/Bmath.hpp>
#include <cpputil/Ptr.hpp>
#include <distributions.hpp>
#include <LinAlg/Matrix.hpp>
#include <LinAlg/Vector.hpp>
#include <stats/DataTable.hpp>
#include <stats/Spline.hpp>
#include <vector>

namespace BOOM {

  enum RuleType {
    ERROR = -1,
    INTERCEPT,
    LINEAR,
    SPLIT,
    SPLINE,
    INTERACTION };

  class BagOfRulesPrior;
  class ModifyRuleProposalBase;
  class ModifyRuleProposalDistributionFactory;

  // This is a base class for constructing rules for basis expansions of
  // observations of covariates.
  class BaseRule
      : private RefCounted
  {
    friend void intrusive_ptr_add_ref(BaseRule *r) { r->up_count(); }
    friend void intrusive_ptr_release(BaseRule *r) {
      r->down_count();
      if (r->ref_count()==0) {
        delete r; }
    }

   public:
    ~BaseRule() override;

    virtual RuleType rule_type() const = 0;

    // A new copy of the rule.
    virtual BaseRule * clone() const = 0;

    // The number of columns in the basis matrix returned by basis().
    virtual int dim() const = 0;

    // Returns the basis representation of the rule as a Matrix, where
    // the columns of the matrix contain the basis elements.
    virtual Matrix basis() const = 0;

    // Returns the basis matrix for the interaction between *this and
    // "rule."
    virtual Matrix interact_with_rule(const BaseRule &rule) const = 0;

    // Methods used to implement interact_with_rule() through
    // double-dispatch.
    virtual Matrix interact_with_intercept_rule(const BaseRule & rule) const;
    virtual Matrix interact_with_linear_rule(const BaseRule & rule) const;
    virtual Matrix interact_with_split_rule(const BaseRule & rule) const;
    virtual Matrix interact_with_spline_rule(const BaseRule & rule) const;
    virtual Matrix interact_with_interaction_rule(const BaseRule & rule) const;


    //---------------------------------------------------------------------
    // Methods for double dispatch.

    // Double dispatch for evaluating prior distributions.
    virtual double evaluate_bag_of_rules_prior(
        const BagOfRulesPrior *prior) const = 0;

    // NOTE: If the set of potential proposal distributions expands,
    // we can change ModifyRuleProposalBase to something more general
    // like ProposalBase.
    //
    // NOTE: return value is a naked pointer, but it should be caught
    // by a Ptr.
    virtual ModifyRuleProposalBase * create_proposal_distribution(
        ModifyRuleProposalDistributionFactory *factory) = 0;

   protected:
    // Generates the matrix column basis elements for the rule. The rule
    // is then stored as a private member.
    virtual void generate_basis(const DataTable &data_table) = 0;
  };

  //============= INTERCEPT RULE ===============
  // A dummy variable of all 1s.
  class InterceptRule :
      virtual public BaseRule
  {
   public:
    // Args:
    //  data_table: the data table containing the observations and
    //    covariate measurements.
    InterceptRule(const DataTable & data_table);

    RuleType rule_type() const override { return INTERCEPT;}

    InterceptRule * clone() const override { return new InterceptRule(*this); }

    // The number of columns in the basis matrix returned by basis().
    int dim() const override;

    // The basis expansion of the rule, which is a column matrix of all
    // 1s.
    Matrix basis() const override;

    double evaluate_bag_of_rules_prior(const BagOfRulesPrior *) const override;
    ModifyRuleProposalBase * create_proposal_distribution(
        ModifyRuleProposalDistributionFactory *factory) override;

   protected:
    // Returns the basis matrix for the interaction between *this and
    // rule.
    Matrix interact_with_rule(const BaseRule & rule) const override;

    // Generates the basis matrix, which is all 1s the size of number of
    // observations.
    void generate_basis(const DataTable & data_table) override;

   private:
    Matrix basis_;
  };

  //============= LINEAR RULE ==================
  // A variable containing the measurements for a specified continuous
  // variable.
  class LinearRule :
      virtual public BaseRule
  {
   public:
    // Args:
    //   variable: the position of the covariate of interest in the
    //     data_table.
    //   data_table: the data table containing the observations and the
    //     covariate measurements.
    //   In the case of a LinearRule, the basis expansion is the column
    //   Matrix containing the measurements of the covariate of
    //   interest.
    LinearRule(int variable, const DataTable &data_table);

    RuleType rule_type() const override { return LINEAR;}

    LinearRule * clone() const override { return new LinearRule(*this); }

    // The number of columns in the basis matrix returned by basis().
    int dim() const override;

    // The basis expansion of the rule, which is a copy of the covariate
    // measurements for variable_.
    Matrix basis() const override;

    // The position in the DataTable of the variable being modelled by
    // this rule.
    int variable() const;

    double evaluate_bag_of_rules_prior(const BagOfRulesPrior *) const override;
    ModifyRuleProposalBase * create_proposal_distribution(
        ModifyRuleProposalDistributionFactory *factory) override;

   private:
    // Returns the basis matrix for the interaction between *this and
    // rule.
    Matrix interact_with_rule(const BaseRule & rule) const override;

    // Generates the basis expansion of the rule for the
    // corresponding variable_ in the data_table.
    void generate_basis(const DataTable &data_table) override;

    // Sets variable_ = variable and changes the basis_
    // correspondingly.
    void generate_basis(int variable, const DataTable &data_table);

    int variable_;
    Matrix basis_;
  };

  //================ SPLITTING RULES =========================
  // A dummy variable indicating whether a continuous variable is above
  // or below a specified splitting point.
  class SplitRule : public BaseRule
  {
   public:
    // Args:
    //  variable: the position of the covariate of interest in the
    //  data_table.
    //  split_point: value to which measurements are compared.
    //  descend_left: if TRUE, then covariate measurements less than
    //  split_point are marked 1 and the others 0. If FALSE,
    //  measurements greater than or equal to.
    //  data_table: the data table holding the observations and the
    //  covariate measurements.
    SplitRule(
        int variable,
        double split_point,
        bool descend_left,
        const DataTable &data_table);

    ~SplitRule() override;

    RuleType rule_type() const override { return SPLIT;}

    SplitRule * clone() const override { return new SplitRule(*this); }

    // The number of columns in the basis matrix returned by basis().
    int dim() const override;

    // The position in the DataTable of the variable being modelled by
    // this rule.
    int variable() const;

    // The basis expansion of the rule with respect to the variable,
    // which in this case is a column of 1s and 0s.
    Matrix basis() const override;

    // Value to which covariate measurements are compared and marked 0
    // or 1, depending on descend_left_.
    double split_point() const;

    // Returns the descend_left_ private member. If true, covariate
    // measurements less than split_point_ are marked 1 and the others
    // 0. If false, vice versa.
    bool descend_left() const;

    double evaluate_bag_of_rules_prior(const BagOfRulesPrior *) const override;

    ModifyRuleProposalBase * create_proposal_distribution(
        ModifyRuleProposalDistributionFactory *factory) override;

   private:
    // Performs basis expansion of the rule for the variable_,
    // split_point_, and descend_left_.
    void generate_basis(const DataTable &data_table) override;

    // Sets the private members variable_, split_point_, descend_left_
    // equal to the corresponding inputs and generates basis_.
    void generate_basis(int variable,
                        double split_point,
                        bool descend_left,
                        const DataTable &data_table);

    // Returns the basis matrix for the interaction of *this with rule.
    Matrix interact_with_rule(const BaseRule & rule) const override;

    int variable_;
    Matrix basis_;
    double split_point_;
    bool descend_left_;

    // Called within generate_basis(const DataTable &data_table) if
    // descend_left_ = true. Returns a vector with 1s for observations
    // LESS than split point and 0s otherwise.
    Vector rules_expand_descend_left(const Vector &v,
                                     double split_point) const;

    // Called within generate_basis(const DataTable &data_table) if
    // descend_left_ = false. Returns a vector with 1s for observations
    // GREATER THAN OR EQUAL TO split point and 0s otherwise.
    Vector rules_expand_descend_right(const Vector &v,
                                      double split_point) const;
  };

  //============== SPLINE ====================
  // A spline basis expansion for a specific covariate.
  class SplineRule : public BaseRule
  {
   public:
    // Args:
    //  variable: the position of the covariate of interest in the
    //     data_table.
    //  knots: a vector of which defines the positions of the
    //   knots for the spline basis expansion for the covariate indexed
    //   by variable in the data_table.
    //  order: the order of the spline basis expansion to be used.
    //  data_table: the data table containing the observations and the
    //   covariate measurements.
    // NOTE: If order_ = k, then the number of knots must be at least
    // k+4. The basis_ Matrix is of size n x order_, where n is the
    // number of observations. Each column is the corresponding basis
    // function element, evaluated at the observed covariate value.
    SplineRule(int variable,
               Vector knots,
               int order,
               const DataTable &data_table);

    ~SplineRule() override;

    RuleType rule_type() const override { return SPLINE;}

    SplineRule * clone() const override { return new SplineRule(*this); }

    // The number of columns in the basis matrix returned by basis().
    int dim() const override;

    // The variable index in the data table.
    int variable() const;

    // The basis expansion of the rule. Each column in the basis
    // expansion corresponds to a basis element, and each value in a
    // column is the basis element evaluated at the corresponding
    // covariate observation.
    Matrix basis() const override;

    // The positions of the knots in the basis spline expansion.
    Vector knots() const;

    // The order of the spline basis expansion.
    int order() const;

    double evaluate_bag_of_rules_prior(const BagOfRulesPrior *) const override;

    ModifyRuleProposalBase * create_proposal_distribution(
        ModifyRuleProposalDistributionFactory *factory) override;

   private:
    // This function generates the basis_ expansion of the rule for the
    // variable_, knots_, and order_.
    void generate_basis(const DataTable &data_table) override;

    // This function sets the private members to the corresponding
    // inputs and generates the basis expansion.
    void generate_basis(
        int variable,
        Vector knots,
        int order,
        const DataTable &data_table);

    // Returns the interaction matrix of *this with rule, which is
    // composed of the pairwise products of the column elements of the
    // bases of the two rules.
    Matrix interact_with_rule(const BaseRule & rule) const override;

    int variable_;
    Matrix basis_;
    Vector knots_;
    int order_;
  };

  //============= INTERACTION ================
  // The basis expansion of the interaction of two rules.
  class InteractionRule : public BaseRule
  {
   public:
    // Args:
    //  child_rules: vector of rules to be interacted.
    //  data_table: the data table containing the observations and the
    //  covariate measurements.

    // TODO(user): The InteractionRule still has the tree structure,
    // although it is no longer binary. The child_rules_ vector may
    // currently look like the following:
    //
    //             ____________ child_rules_  ______________
    //            /            /            \                \
    //           /            /              \                \
    //    some_rule_1, some_rule_2, interaction_rule_3, ... some_rule_K
    //                                     |
    //                                     |
    //                                child_rules_
    //                               /  /  | \  \ \
    //                               .  .  .  .  . .
    //
    // We need to collapse this tree structure, because if we cannot
    // remove individual children of interaction_rule_3 from the
    // interaction.
    InteractionRule(std::vector<Ptr<BaseRule>> child_rules,
                   const DataTable &data_table);

    ~InteractionRule() override;

    RuleType rule_type() const override {return INTERACTION;}

    InteractionRule * clone() const override {
      return new InteractionRule(*this); }

    // The number of columns in the basis matrix returned by basis().
    int dim() const override;

    // The basis expansion of the interaction of the rules in child_rules_.
    Matrix basis() const override;

    // Returns the vector of pointers to child rules.
    std::vector<Ptr<BaseRule>> child_rules() const;

    void set_child_rule(Ptr<BaseRule> child_rule, int which_position);

    // Returns the basis matrix for the interaction between *this and "rule."
    Matrix interact_with_rule(const BaseRule & rule) const override;

    double evaluate_bag_of_rules_prior(const BagOfRulesPrior *) const override;

    ModifyRuleProposalBase * create_proposal_distribution(
        ModifyRuleProposalDistributionFactory *factory) override;

    // Performs the basis expansion via double dispatch
    // between the two interacting rules.
    void generate_basis(const DataTable & data_table) override;

   private:
    Matrix basis_;
    std::vector<Ptr<BaseRule>> child_rules_;
    const DataTable *data_table_;

    // Returns the interaction matrix for the the two rules being
    // interacted.
    Matrix interact_two_rules(const BaseRule & rule_1,
                              const BaseRule & rule_2) const;
    // Returns the basis matrix for the interaction of all rules in the
    // std::vector.
    Matrix interact_multiple_rules(
        const std::vector<Ptr<BaseRule>> rules,
        const DataTable &data_table) const;
  };

  namespace Rules {
    // Returns 'false' if rule generates a less than full rank basis
    // matrix.  If this function returns 'true' then the rule is a
    // legal rule.
    bool check_rule(const BaseRule &rule);
  }  // namespace Rules

}  // namespace BOOM

#endif // BOOM_NONPARAMETRIC_BAYES_RULES_HPP_
