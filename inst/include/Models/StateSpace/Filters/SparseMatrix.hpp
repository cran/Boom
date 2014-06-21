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

#ifndef BOOM_SPARSE_MATRIX_HPP_
#define BOOM_SPARSE_MATRIX_HPP_

#include <map>
#include <boost/shared_ptr.hpp>

#include <LinAlg/Vector.hpp>
#include <LinAlg/Matrix.hpp>
#include <LinAlg/SpdMatrix.hpp>
#include <LinAlg/SubMatrix.hpp>
#include <LinAlg/Types.hpp>

#include <Models/ParamTypes.hpp>

#include <cpputil/ThrowException.hpp>
#include <cpputil/RefCounted.hpp>
#include <cpputil/Ptr.hpp>
#include <cpputil/report_error.hpp>

#include <Models/StateSpace/Filters/SparseVector.hpp>

namespace BOOM{
  //======================================================================
  // The SparseMatrixBlock classes are designed to be used as elements
  // in a BlockDiagonalMatrix.  They only need to implement the
  // functionality required by the DiagonalMatrix implementation.
  class SparseMatrixBlock : private RefCounted {
   public:
    virtual ~SparseMatrixBlock(){}
    virtual SparseMatrixBlock *clone()const=0;

    virtual int nrow()const=0;
    virtual int ncol()const=0;

    // lhs = this * rhs
    virtual void multiply(VectorView lhs, const ConstVectorView &rhs)const=0;

    // lhs = this.transpose() * rhs
    virtual void Tmult(VectorView lhs, const ConstVectorView &rhs)const = 0;

    // Replace x with this * x.  Assumes *this is square.
    virtual void multiply_inplace(VectorView x)const = 0;

    // m = this * m
    virtual void matrix_multiply_inplace(SubMatrix m)const;

    // m = m * this->t();
    virtual void matrix_transpose_premultiply_inplace(SubMatrix m)const;

    // Add *this to block
    // TODO(stevescott):  needs unit tests for all derived classes
    virtual void add_to(SubMatrix block)const = 0;
    void conforms_to_rows(int i)const;
    void conforms_to_cols(int i)const;
    void check_can_add(const SubMatrix &m)const;
    virtual Mat dense()const;
    int ref_count()const{return RefCounted::ref_count();}
   private:
    friend void intrusive_ptr_add_ref(SparseMatrixBlock *m){m->up_count();}
    friend void intrusive_ptr_release(SparseMatrixBlock *m){
      m->down_count();
      if(m->ref_count()==0) delete m;}
  };

  //======================================================================
  // The LocalLinearTrendMatrix is
  //  1 1
  //  0 1
  //  It corresponds to state elements [mu, delta], where mu[t] =
  //  mu[t-1] + delta[t-1] + error[0] and de[ta[t] = delta[t-1] +
  //  error[1].
  class LocalLinearTrendMatrix : public SparseMatrixBlock {
   public:
    virtual LocalLinearTrendMatrix * clone()const;
    virtual int nrow()const{return 2;}
    virtual int ncol()const{return 2;}
    virtual void multiply(VectorView lhs, const ConstVectorView &rhs)const;
    virtual void Tmult(VectorView lhs, const ConstVectorView &rhs)const;
    virtual void multiply_inplace(VectorView x)const;
    virtual void add_to(SubMatrix block)const;
    virtual Mat dense()const;
  };

  //======================================================================
  // A SparseMatrixBlock filled with a DenseMatrix.  I.e. a dense
  // sub-block of a sparse matrix.
  class DenseMatrix : public SparseMatrixBlock {
   public:
    DenseMatrix(const Mat &m) : m_(m){}
    DenseMatrix(const DenseMatrix &rhs)
        : SparseMatrixBlock(rhs),
          m_(rhs.m_)
    {}
    virtual DenseMatrix * clone()const{return new DenseMatrix(*this);}
    int nrow()const{return m_.nrow();}
    int ncol()const{return m_.ncol();}
    void multiply(VectorView lhs, const ConstVectorView &rhs)const{
      lhs = m_ * rhs; }
    void Tmult(VectorView lhs, const ConstVectorView &rhs)const{
      lhs = m_.Tmult(rhs); }
    void multiply_inplace(VectorView x)const{ x = m_ * x;}
    void add_to(SubMatrix block)const{ block += m_; }
    virtual Mat dense()const{ return m_; }
   private:
    Mat m_;
  };

  //======================================================================
  // A SparseMatrixBlock filled with a dense SpdMatrix.
  class DenseSpd : public SparseMatrixBlock {
   public:
    DenseSpd(const Spd &m) : m_(m){}
    DenseSpd(const DenseSpd &rhs) : SparseMatrixBlock(rhs), m_(rhs.m_){}
    virtual DenseSpd * clone()const{return new DenseSpd(*this);}
    void set_matrix(const Spd &m){m_ = m;}
    int nrow()const{return m_.nrow();}
    int ncol()const{return m_.ncol();}
    void multiply(VectorView lhs, const ConstVectorView &rhs)const{
      lhs = m_ * rhs; }
    void Tmult(VectorView lhs, const ConstVectorView &rhs)const{
      lhs = m_ * rhs; }
    void multiply_inplace(VectorView x)const{ x = m_ * x;}
    void add_to(SubMatrix block)const{ block += m_; }
   private:
    Spd m_;
  };

  //======================================================================
  // A component that is is a diagonal matrix (a square matrix with
  // zero off-diagonal components).  The diagonal elements can be
  // changed to arbitrary values after construction.  This class is
  // conceptutally similar to UpperLeftDiagonalMatrix, but it allows
  // different behavior with respect to setting its elements to
  // arbitrary values.
  class DiagonalMatrixBlock : public SparseMatrixBlock {
   public:
    DiagonalMatrixBlock(int size)
        : diagonal_elements_(size)
    {}
    DiagonalMatrixBlock(const Vec &diagonal_elements)
        : diagonal_elements_(diagonal_elements)
    {}
    DiagonalMatrixBlock * clone()const{return new DiagonalMatrixBlock(*this);}
    void set_elements(const Vec &v){diagonal_elements_ = v;}
    void set_elements(const VectorView &v){diagonal_elements_ = v;}
    void set_elements(const ConstVectorView &v){diagonal_elements_ = v;}
    double operator[](int i)const{return diagonal_elements_[i];}
    double & operator[](int i){return diagonal_elements_[i];}
    virtual int nrow()const{return diagonal_elements_.size();}
    virtual int ncol()const{return diagonal_elements_.size();}
    virtual void multiply(VectorView lhs, const ConstVectorView &rhs)const{
      lhs = diagonal_elements_;
      lhs *= rhs;
    }
    virtual void Tmult(VectorView lhs, const ConstVectorView &rhs)const{
      multiply(lhs, rhs);
    }
    virtual void multiply_inplace(VectorView x)const{x *= diagonal_elements_;}
    virtual void matrix_multiply_inplace(SubMatrix m)const{
      for(int i = 0; i < m.ncol(); ++i){
        m.col(i) *= diagonal_elements_;
      }
    }

    virtual void matrix_transpose_premultiply_inplace(SubMatrix m)const{
      for(int i = 0; i < m.nrow(); ++i){
        m.row(i) *= diagonal_elements_;
      }
    }

    void add_to(SubMatrix block)const{
      block.diag() += diagonal_elements_;
    }

   private:
    Vec diagonal_elements_;
  };

  //======================================================================
  // A seasonal state space matrix describes the state evolution in an
  // dynamic linear model.  Conceptually it looks like this:
  // -1 -1 -1 -1 ... -1
  //  1  0  0  0 .... 0
  //  0  1  0  0 .... 0
  //  0  0  1  0 .... 0
  //  0  0  0  1 .... 0
  // A row of -1's at the top, then an identity matrix with a column
  // of 0's appended on the right hand side.
  class SeasonalStateSpaceMatrix : public SparseMatrixBlock {
   public:
    SeasonalStateSpaceMatrix(int number_of_seasons);
    SeasonalStateSpaceMatrix * clone()const;
    virtual int nrow()const;
    virtual int ncol()const;

    // lhs = (*this) * rhs;
    virtual void multiply(VectorView lhs, const ConstVectorView &rhs)const;
    // lhs = this->transpose() * rhs
    virtual void Tmult(VectorView lhs, const ConstVectorView &rhs)const;
    // x = (*this) * x;
    virtual void multiply_inplace(VectorView x)const;
    virtual void add_to(SubMatrix block)const;
    virtual Mat dense()const;
   private:
    int number_of_seasons_;
  };

  //======================================================================
  // An AutoRegressionTransitionMatrix is a [p X p] matrix with top
  // row containing a vector of autoregression parameters.  The lower
  // left block is a [p-1 X p-1] identity matrix (i.e. a shift-down
  // operator), and the lower right block is a [p-1 X 1] vector of
  // 0's.
  class AutoRegressionTransitionMatrix : public SparseMatrixBlock{
   public:
    AutoRegressionTransitionMatrix(Ptr<VectorParams> rho);
    AutoRegressionTransitionMatrix(const AutoRegressionTransitionMatrix &rhs);
    AutoRegressionTransitionMatrix * clone()const;

    virtual int nrow()const;
    virtual int ncol()const;
    // lhs = this * rhs
    virtual void multiply(VectorView lhs, const ConstVectorView &rhs)const;
    virtual void Tmult(VectorView lhs, const ConstVectorView &rhs)const;
    virtual void multiply_inplace(VectorView x)const;
    virtual void add_to(SubMatrix block)const;
    // virtual void matrix_multiply_inplace(SubMatrix m)const;
    // virtual void matrix_transpose_premultiply_inplace(SubMatrix m)const;
    virtual Mat dense()const;
   private:
    Ptr<VectorParams> autoregression_params_;
  };

  //======================================================================
  // The [dim x dim] identity matrix
  class IdentityMatrix : public SparseMatrixBlock{
   public:
    IdentityMatrix(int dim) : dim_(dim){}
    virtual IdentityMatrix * clone()const{return new IdentityMatrix(*this);}
    virtual int nrow()const{return dim_;}
    virtual int ncol()const{return dim_;}
    virtual void multiply(VectorView lhs, const ConstVectorView &rhs)const{
      conforms_to_cols(rhs.size());
      conforms_to_rows(lhs.size());
      lhs = rhs;}
    virtual void Tmult(VectorView lhs, const ConstVectorView &rhs)const{
      conforms_to_rows(rhs.size());
      conforms_to_cols(lhs.size());
      lhs = rhs;}
    virtual void multiply_inplace(VectorView x)const{}
    virtual void matrix_multiply_inplace(SubMatrix m)const{}
    virtual void matrix_transpose_premultiply_inplace(SubMatrix m)const{}
    virtual void add_to(SubMatrix block)const{ block.diag() += 1.0; }
   private:
    int dim_;
  };

  //======================================================================
  // A scalar constant times the identity matrix
  class ConstantMatrix : public SparseMatrixBlock{
   public:
    ConstantMatrix(int dim, double value)
        : dim_(dim),
          value_(value)
    {}
    virtual ConstantMatrix * clone()const{return new ConstantMatrix(*this);}
    virtual int nrow()const{return dim_;}
    virtual int ncol()const{return dim_;}
    virtual void multiply(VectorView lhs, const ConstVectorView &rhs)const{
      conforms_to_cols(rhs.size());
      conforms_to_rows(lhs.size());
      // Doing this operation in two steps, insted of lhs = rhs *
      // value_, eliminates a temporary that profiliing found to be
      // expensive.
      lhs = rhs;
      lhs *= value_;
    }
    virtual void Tmult(VectorView lhs, const ConstVectorView &rhs)const{
      conforms_to_rows(rhs.size());
      conforms_to_cols(lhs.size());
      lhs = rhs * value_;
    }
    virtual void multiply_inplace(VectorView x)const{
      x *= value_;}
    virtual void matrix_multiply_inplace(SubMatrix x)const{
      x *= value_;}
    virtual void matrix_transpose_premultiply_inplace(SubMatrix x)const{
      x *= value_;}
    virtual void add_to(SubMatrix block)const{block.diag() += value_;}
    void set_value(double value){value_ = value;}
   private:
    int dim_;
    double value_;
  };

  //======================================================================
  // A square matrix of all zeros.
  class ZeroMatrix : public ConstantMatrix{
   public:
    ZeroMatrix(int dim) : ConstantMatrix(dim, 0.0){}
    virtual ZeroMatrix * clone()const{return new ZeroMatrix(*this);}
    virtual void add_to(SubMatrix block)const{}
  };

  //======================================================================
  //  A matrix that is all zeros except for a single nonzero value in
  //  the (0,0) corner.
  class UpperLeftCornerMatrix : public SparseMatrixBlock {
   public:
    UpperLeftCornerMatrix(int dim, double value)
        : dim_(dim),
          value_(value)
    {}
    virtual UpperLeftCornerMatrix * clone()const{
      return new UpperLeftCornerMatrix(*this);}
    int nrow()const{return dim_;}
    int ncol()const{return dim_;}
    virtual void multiply(VectorView lhs, const ConstVectorView &rhs)const{
      conforms_to_cols(rhs.size());
      conforms_to_rows(lhs.size());
      lhs = rhs * 0;
      lhs[0] = rhs[0] * value_;
    }
    virtual void Tmult(VectorView lhs, const ConstVectorView &rhs)const{
      // An upper left corner matrix is symmetric, so Tmult is the
      // same as multiply.
      multiply(lhs, rhs); }
    virtual void multiply_inplace(VectorView x)const{
      double tmp = x[0];
      x = 0;
      x[0] = tmp * value_;
    }
    void set_value(double value){value_ = value;}
    void add_to(SubMatrix block)const{ block(0,0) += value_; }
   private:
    int dim_;
    double value_; // the value in the upper left corner of the matrix
  };

  //======================================================================
  // A diagonal matrix that is zero in all but (at most) one element.
  class SingleSparseDiagonalElementMatrix : public SparseMatrixBlock{
   public:
    SingleSparseDiagonalElementMatrix(int dim, double value, int which_element)
        : dim_(dim),
          value_(value),
          which_element_(which_element)
    {}
    virtual SingleSparseDiagonalElementMatrix * clone()const{
      return new SingleSparseDiagonalElementMatrix(*this);}

    void set_value(double value){value_ = value;}
    void set_element(int which_element){which_element_ = which_element;}

    virtual int nrow()const{return dim_;}
    virtual int ncol()const{return dim_;}
    virtual void multiply(VectorView lhs, const ConstVectorView &rhs)const{
      conforms_to_rows(lhs.size());
      conforms_to_cols(rhs.size());
      lhs = 0;
      lhs[which_element_] = value_ * rhs[which_element_];
    }
    virtual void Tmult(VectorView lhs, const ConstVectorView &rhs)const{
      // Symmetric
      multiply(lhs, rhs);
    }
    virtual void multiply_inplace(VectorView x)const{
      conforms_to_cols(x.size());
      x[which_element_] *= value_;
    }
    virtual void add_to(SubMatrix block)const{
      check_can_add(block);
      block(which_element_, which_element_) += value_;
    }
   private:
    int dim_;
    double value_;
    int which_element_;
  };

  //======================================================================
  // A diagonal matrix whose diagonal entries are zero beyond a
  // certain point.  Diagonal entry i is the product of a
  // BOOM::UnivParams and a constant scalar factor.  Interesting
  // special cases that can be handled include
  //  *) The entire diagonal is nonzero.
  //  *) All scale factors are 1.
  class UpperLeftDiagonalMatrix : public SparseMatrixBlock {
   public:
    UpperLeftDiagonalMatrix(const std::vector<Ptr<UnivParams> > &diagonal,
                            int dim)
        : diagonal_(diagonal),
          dim_(dim),
          constant_scale_factor_(diagonal.size(), 1.0)
    {
      check_diagonal_dimension(dim_, diagonal_);
      check_scale_factor_dimension(diagonal, constant_scale_factor_);
    }

    UpperLeftDiagonalMatrix(const std::vector<Ptr<UnivParams> > &diagonal,
                            int dim,
                            const Vector &scale_factor)
        : diagonal_(diagonal),
          dim_(dim),
          constant_scale_factor_(scale_factor)
    {
      check_diagonal_dimension(dim_, diagonal_);
      check_scale_factor_dimension(diagonal_, constant_scale_factor_);
    }

    virtual UpperLeftDiagonalMatrix * clone()const {
      return new UpperLeftDiagonalMatrix(*this);}
    virtual int nrow()const{return dim_;};
    virtual int ncol()const{return dim_;}
    virtual void multiply(VectorView lhs, const ConstVectorView &rhs)const{
      conforms_to_cols(rhs.size());
      conforms_to_rows(lhs.size());
      for(int i = 0; i < diagonal_.size(); ++i){
        lhs[i] = rhs[i] * diagonal_[i]->value() * constant_scale_factor_[i];
      }
      for(int i = diagonal_.size(); i < dim_; ++i) lhs[i] = 0;
    }
    virtual void Tmult(VectorView lhs, const ConstVectorView &rhs)const{
      multiply(lhs, rhs);
    }
    virtual void multiply_inplace(VectorView x)const{
      conforms_to_cols(x.size());
      for(int i = 0; i < diagonal_.size(); ++i){
        x[i] *= diagonal_[i]->value() * constant_scale_factor_[i];
      }
      for(int i = diagonal_.size(); i < dim_; ++i) x[i] = 0;
    }

    virtual void add_to(SubMatrix block)const{
      conforms_to_rows(block.nrow());
      conforms_to_cols(block.ncol());
      for(int i = 0; i < diagonal_.size(); ++i){
        block(i,i) += diagonal_[i]->value() * constant_scale_factor_[i];
      }
    }
   private:
    std::vector<Ptr<UnivParams> > diagonal_;
    int dim_;
    Vector constant_scale_factor_;

    void check_diagonal_dimension(
        int dim, const std::vector<Ptr<UnivParams> > &diagonal){
      if(dim < diagonal.size()){
        report_error("dim must be at least as large as diagonal in "
                     "constructor for UpperLeftDiagonalMatrix");
      }
    }

    void check_scale_factor_dimension(
        const std::vector<Ptr<UnivParams> > &diagonal,
        const Vec &scale_factor){
      if(diagonal.size() != scale_factor.size()){
        report_error("diagonal and scale_factor must be the same size in "
                     "constructor for UpperLeftDiagonalMatrix");
      }
    }
  };

  //======================================================================
  // A SparseKalmanMatrix is a sparse matrix that can be used in the
  // Kalman recursions.  This may get expanded to a more full fledged
  // sparse matrix class later on, if need be.
  class SparseKalmanMatrix {
   public:
    virtual ~SparseKalmanMatrix(){}

    virtual int nrow()const=0;
    virtual int ncol()const=0;

    virtual Vec operator*(const Vec &v)const = 0;
    virtual Vec operator*(const VectorView &v)const = 0;
    virtual Vec operator*(const ConstVectorView &v)const = 0;

    virtual Vec Tmult(const Vec &v)const=0;
    // P -> this * P * this.transpose()
    virtual void sandwich_inplace(Spd &P)const;
    virtual void sandwich_inplace_submatrix(SubMatrix P)const;

    // P += *this
    virtual Mat & add_to(Mat &P)const = 0;
    virtual SubMatrix add_to_submatrix(SubMatrix P)const;

    // Returns a dense matrix representation of *this.  Mainly for
    // debugging and testing.
    Mat dense()const;
  };

  //======================================================================
  // The state transition equation for a dynamic linear model will
  // typically involve a block diagonal matrix.  The blocks will
  // typically be:  SeasonalStateSpaceMatrix, IdentityMatrix, etc.
  class BlockDiagonalMatrix : public SparseKalmanMatrix{
   public:
    // Start off with an empty matrix.  Use add_block() to add blocks
    // Adds a block to the block diagonal matrix
    BlockDiagonalMatrix();
    void add_block(Ptr<SparseMatrixBlock> m);
    void replace_block(int which_block, Ptr<SparseMatrixBlock> b);
    void clear();

    virtual int nrow()const;
    virtual int ncol()const;

    virtual Vec operator*(const Vec &v)const;
    virtual Vec operator*(const VectorView &v)const;
    virtual Vec operator*(const ConstVectorView &v)const;

    Vec Tmult(const Vec &r)const;
    // P -> this * P * this.transpose()
    virtual void sandwich_inplace(Spd &P)const;
    virtual void sandwich_inplace_submatrix(SubMatrix P)const;

    // sandwich(P) = this * P * this.transpose()
    Spd sandwich(const Spd &P)const;

    virtual Mat & add_to(Mat &P)const;
    virtual SubMatrix add_to_submatrix(SubMatrix P)const;
   private:
    // Replace middle with left * middle * right.transpose()
    void sandwich_inplace_block(const SparseMatrixBlock &left,
                                const SparseMatrixBlock &right,
                                SubMatrix middle)const;

    // Returns the (i,j) block of the matrix m, with block sizes
    // determined by the rows and columns of the entries in blocks_.
    SubMatrix get_block(Mat &m, int i, int j)const;
    SubMatrix get_row_block(Mat &m, int block)const;
    SubMatrix get_col_block(Mat &m, int block)const;
    SubMatrix get_submatrix_block(SubMatrix m, int i, int j)const;
    std::vector<Ptr<SparseMatrixBlock> > blocks_;

    int nrow_;
    int ncol_;

    // row_boundaries_[i] contains the one-past-the-end position of
    std::vector<int> row_boundaries_;
    std::vector<int> col_boundaries_;
  };
  //======================================================================

  Vec operator*(const SparseMatrixBlock &,
                const Vec &);

  // P += TPK * K.transpose * w
  void add_outer_product(Spd &P, const Vec &TPK, const Vec &K, double w);

  // P += RQR
  void add_block_diagonal(Spd &P, const BlockDiagonalMatrix &RQR);

}
#endif // BOOM_SPARSE_MATRIX_HPP_
