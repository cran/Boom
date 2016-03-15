\name{normal.prior}
\title{Normal (scalar Gaussian) prior distribution}
\alias{NormalPrior}

\description{Specifies a scalar Gaussian prior distribution.}

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
