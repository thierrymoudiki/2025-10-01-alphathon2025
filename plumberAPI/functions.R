library(plumber)
library(ahead)
library(misc)

best_params <- readRDS(file="best_params_with_clustering.rds")
params <- list()
params$nb_hidden <- floor(best_params$best_param[1])
params$lags <- floor(best_params$best_param[2])
params$lambda_1 <- 10**best_params$best_param[3]
params$lambda_2 <- 10**best_params$best_param[4]
params$centers <- floor(best_params$best_param[6])

forecast_method <- function(ticker, 
                            n_ahead = 14L, 
                            test_time_index=1) {
  stock_data <- tail(EuStockMarkets[, ticker], 501L)
  returns_stock_data <- diff(log(stock_data))
  time_indices <- time(returns_stock_data)
  ts_splits <- misc::splitts(returns_stock_data, 
                             split_prob = 0.9)
  training_set <- ts_splits$training
  test_set <- ts_splits$testing
  test_time_indices <- seq_along(test_set)
  stopifnot(test_time_index <= max(test_time_indices))
  train <- c(training_set, test_set[seq_len(test_time_index)])
  train_mean <- mean(train)
  train_sd <- sd(train)
  scaled_train <- (train - train_mean) / train_sd
  stopifnot(n_ahead <= 30L)
  level <- 99
  fit_ridge2f <- try(ahead::ridge2f(
    y = scaled_train,
    h = n_ahead,
    nb_hidden = params$nb_hidden,
    lags = min(params$lags, length(scaled_train) - 1L),
    lambda_1 = params$lambda_1,
    lambda_2 = params$lambda_2,
    centers = params$centers,
    level = level,
    B = 250L,
    type_pi = "movingblockbootstrap",
    show_progress = FALSE
  ), silent = FALSE)
  if (!inherits(fit_ridge2f, "try-error")) {
    out <- list()
    out$ticker <- as.character(ticker)
    out$x <- train
    out$level <- as.numeric(level)
    out$method <- "ridge2f"
    #out$model <- fit_ridge2f # add this in the client 
    out$mean <- fit_ridge2f$mean * train_sd + train_mean
    out$lower <- fit_ridge2f$lower * train_sd + train_mean
    out$upper <- fit_ridge2f$upper * train_sd + train_mean
    out$sims <- do.call(cbind, fit_ridge2f$sims) * train_sd + train_mean
    #class(out) <- "forecast" # add this in the client 
    return(out)
  } else {
    list(error = "forecasting failed!")
  }
}
forecast_method <- compiler::cmpfun(forecast_method)