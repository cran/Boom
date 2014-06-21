% Copyright 2011 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{normal.prior}
\title{Normal (scalar Gaussian) prior distribution}
\alias{NormalPrior}

\description{Specifies an inverse Gamma prior for a variance parameter,
  but inputs are defined in terms of a standard deviation.}

\usage{
NormalPrior(mu, sigma, initial.value = mu)
}

\arguments{
  \item{mu}{The mean of the prior distribution.}

  \item{sigma}{The standard deviation of the prior distribution.}

  \item{initial.value}{The initial value of the parameter being modeled
    in the MCMC algorithm.}

}

\references{
  Gelman, Carlin, Stern, Rubin (2003), "Bayesian Data Analysis", Chapman
  and Hall.
}

\author{
  Steven L. Scott \email{stevescott@google.com}
}
