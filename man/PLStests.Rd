% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/PLStests.R
\name{PLStests}
\alias{PLStests}
\title{Model checking for high dimensional generalized linear models based on random projections}
\usage{
PLStests(y, x, family, b0 = 2, np = 10)
}
\arguments{
\item{y}{: y Input matrix with \code{n} rows, 1-dimensional response vector}

\item{x}{: x Input matrix with \code{n} rows, each a \code{p}-dimensional observation vector.}

\item{family}{: Must be "gaussian" or "binomial" for linear or logistic regression model.}

\item{b0}{: a paramter to set bindwith, the default value may better for real data analysing.}

\item{np}{: the number of random projections.}
}
\value{
a list with five parameters returned. \code{h} stand for \code{b_0}.
T_alpha: the p value of our statistics by random projection. T_beta: the p value of our statistic by
 estimated projection. T_cauchy and T_hmp are p value of two combinational method proposed by
 Liu and Xie (2020) and Wilson (2019) respectively. each method combines p values of \code{np} random
 projections. when the estimated projection is zero, the value set be NA.
}
\description{
The function can test goodness-of-fit of a low- or high-dimensional
generalized linear model (GLM) by detecting the presence of nonlinearity in
the conditional mean function of y given x using the statistics proposed by paper xx.
The outputs are p-value of  statisitics.
}
\examples{

set.seed(100)
data("sonar_mines")
x = sonar_mines[,-1]
y = sonar_mines$y

## make y as 0 or 1 for logistic regression
class1 = "R"
class2 ="M"
y = as.character(y)
y[y==class1]=1
y[y==class2]=0
y = as.numeric(y)
y = matrix(y,ncol = 1)

## scale x  and make data to be matrix
data_test_x = x
data_test_x = as.matrix(data_test_x)
data_test_y = as.matrix(y)
data_test_x = scale(data_test_x)
PLStests(data_test_y,data_test_x,family="binomial")


}
\references{
Chen, W., Liu, J., Peng, H., Tan, F., & Zhu, L. (2024). Model checking for high dimensional generalized linear models based on random projections. arXiv [Stat.ME]. Retrieved from http://arxiv.org/abs/2412.10721
}
