\name{plot.many.ts}
\Rdversion{1.1}
\alias{plot.many.ts}
\alias{PlotManyTs}
\title{
  Multiple time series plots
}
\description{
  Plots many time series plots on the same graphical device.  Each plot
  gets its own frame.  Scales can be adjusted to see variation in each
  plot (each plot gets its own scale), or variation between plots
  (common scale).
}
\usage{
PlotManyTs(x, type = "l", gap = 0, boxes = TRUE, truth = NULL,
           thin = 1, labs, same.scale = TRUE, ylim = NULL,
           refline = NULL, color = NULL, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
    matrix or 3-dimensional array to be plotted
}
  \item{type}{
    type of line plots to produce.  See \code{\link{plot.default}} for
    other options.
}
  \item{gap}{
    number of lines of space to put between plots.
}
  \item{boxes}{
    logical.  Should boxes be drawn around each plot?
}
  \item{truth}{
    A vector or matrix of reference values to be added to each plot as a
    horizontal line.  The dimension should match \code{dim(x)[-1]}
}
  \item{thin}{
    Frequency of observations to plot.  E.g. \code{thin=10} means plot
    every 10'th observation.  Thinning can speed things up when plotting large
    amounts MCMC output.
}
  \item{labs}{
    (optional) character vector giving the title (e.g. variable name)
    for each plot.
}
  \item{same.scale}{
    logical.  If true then plots are drawn with a common vertical axis,
    which is displayed on alternating rows of the plot.  If false each
    plot gets its own axis, which is not displayed.
}
  \item{ylim}{

}
  \item{refline}{
    a vector or scalar value to use as a reference line.  This is a
  supplement to the \code{truth} argument.  It can be
  useful when comparing true values (used in a simulation), estimated
  values (e.g. point estimates of parameters) and MCMC output.
}
  \item{color}{
    Vector of colors to use in the plots.
}
  \item{\dots}{
    extra arguments to be passed to \code{\link{plot}} and \code{\link{axis}}.
}
}
\value{
  Called for its side effect.
}
\author{
  Steven L. Scott
}

\seealso{
  \code{\link{plot.ts}} (for plotting a small number of time series)
  \code{\link{plot.macf}}
}
\examples{
x <- matrix(rnorm(1000), ncol=10)
PlotManyTs(x)

PlotManyTs(x, same = FALSE)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ hplot }
