/*
  Copyright (C) 2005 Steven L. Scott

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

#ifndef BOOM_NUMOPT_HPP
#define BOOM_NUMOPT_HPP

#include <string>
#include <LinAlg/SpdMatrix.hpp>

#include <BOOM.hpp>
#include <boost/function.hpp>
#include <cpputil/report_error.hpp>

namespace BOOM{

  // Optimizers work in terms of arbitrary function objects.
  // TODO(stevescott): Replace these with C++ function templates once
  // C++11 becomes more widely supported.
  typedef boost::function<double(const Vector &) > Target;
  typedef boost::function<double(const Vector &x,
                                 Vector &g) > dTarget;
  typedef boost::function<double(const Vector &x,
                                 Vector &g,
                                 Matrix &H) > d2Target;
  typedef boost::function<double(double) > ScalarTarget;

  enum conj_grad_method{ FletcherReeves, PolakRibiere, BealeSorenson};

  double max_nd0(Vector &x, Target tf);

  bool max_nd1_careful(Vector &x,
                       double &log_function_value,
                       Target tf,
                       dTarget dtf,
                       double eps = 1e-5);
  double max_nd1(Vector &x, Target tf, dTarget dtf, double eps = 1e-5);

  double max_nd2(Vector &x, Vector &g, Matrix &h, Target tf, dTarget dtf,
                 d2Target d2tf, double leps = 1e-5);

  bool max_nd2_careful(Vector &x, Vector &g, Matrix &h, double &max_value,
                       Target tf, dTarget dtf, d2Target d2tf,
                       double leps, string &error_msg);

  double numeric_deriv(const ScalarTarget f, double x);
  double numeric_deriv(const ScalarTarget f, double x,
                       double &dx, double &abs_err);

  Vector numeric_gradient(const Vector &x, Target f, double dx);
  Matrix numeric_hessian(const Vector &x, Target f, double dx);
  Matrix numeric_hessian(const Vector &x, dTarget df, double dx);

  //--------- Methods: Each includes a full interface and an inline
  //--------- function providing a simpler interface

  double nelder_mead_driver(Vector &x, Vector &y,
                            Target f,
                            double abstol,
                            double intol,
                            double alpha, double beta, double gamma,
                            bool trace, int & fncount, int maxit);

  double nelder_mead(Vector &x, Vector &y,
                     Target f,
                     double abstol,
                     double intol,
                     double alpha, double beta, double gamma,
                     bool trace, int & fncount, int maxit);

  double bfgs(Vector &x, Target target,
              dTarget  dtarget,
              int maxit, double abstol, double reltol,
              int &fncount, int & grcount, bool &fail, int trace_freq= -1);

  // Minimize the function f using the conjugate gradient algorithm.
  //
  double conj_grad(Vector &x, Vector &y, Target f,
                   dTarget df, double abstol, double intol,
                   conj_grad_method type, bool trace,
                   int &fcnt, int &gcnt, int maxit);


  // Minimize the function f using the Newton-Raphson algorithm.
  // Args:
  //   x: The argument to f.  Input specifies the starting value for
  //     the optimization algorithm.  Output gives the optimizing
  //     value of x.
  //   g: The gradient, to be computed by f.  This should either be
  //     sized appropriately on input, or f should handle the
  //     resizing.
  //   h:  The hessian, to be computed by f.  This should either be
  //     sized appropriately on input, or f should handle the
  //     resizing.
  //   f:  The function to be minimized.
  //   function_call_count:  output.  No need to initialize this.
  //   eps: The algorithm will converge when the (absolute) change in
  //     function values is less than eps.
  //   happy_ending: If true then the algorithm converged happily.  If
  //     not there was a (potential) problem.
  //   error_message: If happy_ending is false, then the reason will
  //     be written to error_message.  If happy_ending is true then
  //     error_message is not used.
  //
  // Returns:
  //   The value of the function at the conclusion of the algorithm.
  double newton_raphson_min(Vector &x,
                            Vector &g,
                            Matrix &h,
                            d2Target f,
                            int &function_call_count,
                            double eps,
                            bool & happy_ending,
                            string &error_message);

  // Minimize a function using derivative-free simulated annealing.
  // Args:
  //   x: The argument of the function to be optimized.  Input
  //     specifies the initial value for the algorithm.  Output gives
  //     the value that optimizes f.
  //   f:  The function to be minimized.
  //   maxit:  The maximum number of function evaluations.
  //   tmax:  The maximum number of evaluations at each temperature step.
  //   ti: "Temperature increment".  Used to adjust the scale of the
  //     random annealing perturbations.
  //
  // Returns:
  //   On exit x is the (approximate) minimizing value of f, and the
  //   return value is f(x).
  double simulated_annealing(Vector &x,
                             Target f,
                             int maxit,
                             int tmax,
                             double ti);

  //======================================================================
  // Negations:
  //

  // A class to use as a target function when maximizing a function of
  // a single variable.  Minimizing the ScalarNegation of f(x) maximizes
  // f(x).
  class ScalarNegation {
   public:
    ScalarNegation(ScalarTarget f)
        : original_function_(f) {}
    double operator()(double x)const{ return -1 * original_function_(x); }
   private:
    ScalarTarget original_function_;
  };

  // Minimizing a Negate(F) maximizes F, where F is a function of many
  // variables.
  class Negate{
  public:
    Negate(Target F) : f(F){}
    double operator()(const Vector &x)const;
  private:
    Target  f;
  };

  // Use this negation when F has first derivatives.
  class dNegate : public Negate{
  public:
    dNegate(Target F, dTarget dF)
      : Negate(F), df(dF){}
    double operator()(const Vector &x)const{
      return Negate::operator()(x);}
    double operator()(const Vector &x, Vector &g)const;
  private:
    dTarget df;
  };

  // Use this Negation when F has first and second derivatives.
  class d2Negate : public dNegate{
  public:
    d2Negate(Target f, dTarget df, d2Target  d2F)
      : dNegate(f, df), d2f(d2F){}
    double operator()(const Vector &x)const{
      return Negate::operator()(x);}
    double operator()(const Vector &x, Vector &g)const{
      return dNegate::operator()(x,g);}
    double operator()(const Vector &x, Vector &g, Matrix &h)const;
  private:
    d2Target d2f;
  };

}
#endif // BOOM_NUMOPT_HPP
