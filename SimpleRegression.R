# SimpleRegression.R
# Implements simple linear regression from scratch using base R

# 1. Compute descriptive statistics
get_dstats <- function(y, x) {
  n <- length(y)
  mean_x <- mean(x)
  mean_y <- mean(y)
  var_x <- var(x)
  var_y <- var(y)
  cov_xy <- cov(x, y)
  
  return(list(
    n = n,
    mean_x = mean_x,
    mean_y = mean_y,
    var_x = var_x,
    var_y = var_y,
    cov_xy = cov_xy
  ))
}

# 2. Estimate coefficients for beta0 and beta1
estimate_coefficients <- function(dstats) {
  beta1 <- dstats$cov_xy / dstats$var_x
  beta0 <- dstats$mean_y - beta1 * dstats$mean_x
  return(c(beta0, beta1))
}

# 3. Predict y_hat
make_prediction <- function(x, beta) {
  beta[1] + beta[2] * x
}

# 4. Estimate sigma
estimate_sigma <- function(y, y_hat) {
  residuals <- y - y_hat
  sqrt(sum(residuals^2) / (length(y) - 2))
}

# 5. Build regression table
make_regression_table <- function(x, beta, sigma, dstats) {
  n <- dstats$n
  var_x <- dstats$var_x
  mean_x2 <- mean(x^2)
  
  var_b1 <- sigma^2 / ((n - 1) * var_x)
  var_b0 <- mean_x2 * var_b1
  
  stderr <- sqrt(c(var_b0, var_b1))
  tstat <- beta / stderr
  pval <- 2 * pt(-abs(tstat), df = n - 2)
  
  return(data.frame(
    value = beta,
    stderr = stderr,
    tstat = tstat,
    pval = pval
  ))
}

# 6. Computation of model metrics
get_metrics <- function(y, sigma, dstats) {
  r_squared <- (dstats$cov_xy / (sqrt(dstats$var_x) * sqrt(dstats$var_y)))^2
  adj_r_squared <- 1 - (1 - r_squared) * (dstats$n - 1) / (dstats$n - 2)
  F <- (dstats$cov_xy^2 / dstats$var_x) / (sigma^2 / (dstats$n - 2))
  pval <- pf(F, df1 = 1, df2 = dstats$n - 2, lower.tail = FALSE)
  
  return(c(
    r_squared = r_squared,
    adj_r_squared = adj_r_squared,
    F_statistic = F,
    F_pval = pval
  ))
}

# 7. Combine into regression function
simple_regression <- function(y, x) {
  dstats <- get_dstats(y, x)
  beta <- estimate_coefficients(dstats)
  fitted_vals <- make_prediction(x, beta)
  residuals <- y - fitted_vals
  sigma <- estimate_sigma(y, fitted_vals)
  regtable <- make_regression_table(x, beta, sigma, dstats)
  metrics <- get_metrics(y, sigma, dstats)
  
  return(list(
    model = data.frame(y = y, x = x),
    coefficients = beta,
    residuals = residuals,
    fitted.values = fitted_vals,
    regtable = regtable,
    sigma = sigma,
    r_squared = metrics["r_squared"],
    adj_r_squared = metrics["adj_r_squared"],
    F = metrics["F_statistic"],
    F_pval = metrics["F_pval"]
  ))
}

# 8. Displaying result
display_report <- function(model) {
  cat("Simple Regression Report\n")
  cat("========================\n")
  cat("Coefficients:\n")
  print(model$coefficients)
  cat("\nRegression Table:\n")
  print(model$regtable)
  cat(sprintf("\nAdjusted R-squared: %.4f\n", model$adj_r_squared))
  cat(sprintf("F-statistic: %.4f\n", model$F))
  cat(sprintf("P-value of F-statistic: %.4f\n", model$F_pval))
}