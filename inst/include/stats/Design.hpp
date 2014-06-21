#ifndef BOOM_DESIGN_HPP
#define BOOM_DESIGN_HPP
/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#include <vector>
#include <map>
#include <string>
#include <limits>
#include <LinAlg/Matrix.hpp>
#include <LinAlg/Types.hpp>
#include <BOOM.hpp>

namespace BOOM{
  // A design matrix is an ordinary Matrix with variable
  // (column) names and a display() method that prints the variable
  // names over each column when the matrix is streamed
  class DesignMatrix : public Mat {
  public:
    DesignMatrix(const Mat &X);
    DesignMatrix(const Mat &X, const std::vector<string> &vnames);
    DesignMatrix(const Mat &X, const std::vector<string> &vnames,
                  const std::vector<string> &baseline_names);
    DesignMatrix(const DesignMatrix &);

    std::vector<string> &vnames(){return vnames_;}
    const std::vector<string> &vnames()const {return vnames_;}
  private:
    std::vector<string> vnames_;
    std::vector<string> baseline_names_;
  };

  ostream & display(ostream &out, const DesignMatrix &m,
                    int prec=5, uint from=0,
                    uint to= std::numeric_limits<uint>::max());
  ostream & operator<<(ostream &out, const DesignMatrix &m);

  //======================================================================
  // Generates a design matrix consisting of an intercept and dummy
  // variables for all effects up to the specified order.  If order =
  // 0 just the intercept is used.  If order = 1 then main effects are
  // added.  If order = 2 then second order interactions are added.
  // Etc.
  DesignMatrix generate_design_matrix(
      const std::map<string, std::vector<string> > & level_names,
      int interaction_order);

  //======================================================================
  // A Configuration is a sequence of factor levels, represented by a
  // vector of integers, that knows how to generate the next Configuration.
  class Configuration {
  public:
    explicit Configuration(const std::vector<int> &nlevels);
    Configuration(const std::vector<int> &nlevels, const std::vector<int> &levels);
    void next();  // advances this configuration to the next one
    bool done()const;  // one past the end
    int level(int factor) const;
    const std::vector<int> & levels()const;
    bool operator==(const Configuration &rhs)const;
    bool operator!=(const Configuration &rhs)const;
    ostream & print(ostream &out)const;
   private:
    std::vector<int> nlevels_;
    std::vector<int> levels_;
  };

  inline ostream & operator<<(ostream &out, const Configuration &config){
    return config.print(out);}

  //======================================================================
  // An ExperimentStructure records the names of the factors in an
  // experiment, as well as the names of all the levels in each
  // factor.
  class ExperimentStructure {
  public:
    ExperimentStructure(const std::map<string, std::vector<string> > &);
    ExperimentStructure(const std::vector<int> & nlevels);
    int nfactors()const;
    int nlevels(int factor)const;
    const std::vector<int> & nlevels()const;
    int nconfigurations()const;  // number of possible configurations
    const string & level_name(int factor, int level) const;
    std::vector<string> baseline_levels()const;
  private:
    // generate a set of "names" from 0 to n-1
    std::vector<string> generate_names(int n)const;
    std::vector<string>  factor_names_;
    std::vector<std::vector<string> > level_names_;
    std::vector<int> nlevels_;
  };

  //======================================================================
  // A FactorDummy is a functor class that returns 1 when its level
  // for its factor is present in a Configuration.  It is a primitive
  // class used to make Effects, which are used in RowBuilder.
  class FactorDummy {
  public:
    FactorDummy(int factor, int level, const string & name);
    bool eval(const Configuration &config)const;
    bool eval(const std::vector<int> &levels)const;
    const string &name()const;
    bool operator==(const FactorDummy &rhs)const;
    bool operator<(const FactorDummy &rhs)const;
    int factor()const;
  private:
    int factor_;
    int level_;
    string name_;
  };

  //======================================================================
  // An Effect can be an intercept, a main effect, or an interation.
  // It a "functor" class that is the product of zero or more
  // FactorDummy's.
  class Effect {
  public:
    Effect();
    Effect(const FactorDummy &factor);
    Effect(const Effect &first, const Effect &second);
    int order()const;  // number of factor dummies
    void add_factor(const FactorDummy &factor);
    // add_effect adds all the factors in the specified Effect to this Effect
    void add_effect(const Effect &effect);
    bool eval(const Configuration &config)const;
    bool eval(const std::vector<int> &levels)const;
    string name()const;
    bool operator==(const Effect &rhs)const;
    bool operator<(const Effect &rhs)const;
    bool has_factor(const FactorDummy &f)const;
    const FactorDummy & factor(int factor_number)const;
  private:
    // only one factor from the same family is allowed
    std::vector<FactorDummy> factors_;
  };

  inline ostream & operator<<(ostream &out, const Effect &effect){
    out << effect.name();
    return out;
  }
  //======================================================================
  // A RowBuilder converts a Configuration into the corresponding row
  // in a design matrix.
  class RowBuilder {
  public:
    RowBuilder();
    // interaction order = the number of factors involved in the
    // highest order interaction.  I.e. interaction_order = 2 means a
    // two-factor interaction
    RowBuilder(const ExperimentStructure &xp, int interaction_order);
    RowBuilder(const std::vector<int> &nlevels, int interaction_order);
    RowBuilder(const std::map<string, std::vector<string> > &level_names,
               int interaction_order);

    void add_effect(const Effect &e);
    bool has_effect(const Effect &e)const;

    // the number of dummy variables required to represent the main
    // effects in a regression model, not including the intercept
    int number_of_main_effects()const;

    const Effect & effect(const int i)const;
    std::vector<double> build_row(const Configuration &config) const;
    std::vector<double> build_row(const std::vector<int> &levels) const;
    int dim()const;  // return the dimension of the row to be built
    std::vector<string> variable_names()const;
  private:
    std::vector<Effect> effects_;
    void setup(const ExperimentStructure &xp, int order);
  };
}  // namespace BOOM
#endif //BOOM_DESIGN_HPP
