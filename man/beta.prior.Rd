% Copyright 2011 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{beta.prior}
\title{Beta prior for a binomial proportion}
\alias{BetaPrior}

\description{Specifies beta prior distribution for a binomial
  probability parameter.}

\usage{
  BetaPrior(a = 1, b = 1, mean = NULL, sample.size = NULL)
}

\arguments{
  \item{a}{A positive real number interpretable as a prior success count.}
  \item{b}{A positive real number interpretable as a prior failure
    count.}
  \item{mean}{A positive real number representing \code{a/(a+b)}.}
  \item{sample.size}{A positive real number representing \code{a+b}.}
}

\details{
  The distribution should be specified either with \code{a} and
  \code{b}, or with \code{mean} and \code{sample.size}.
}

\references{
  Gelman, Carlin, Stern, Rubin (2003), "Bayesian Data Analysis", Chapman
  and Hall.
}

\author{
  Steven L. Scott \email{stevescott@google.com}
}
