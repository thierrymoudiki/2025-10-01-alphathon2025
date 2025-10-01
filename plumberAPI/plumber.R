source("functions.R")

#* @apiTitle Serving ahead::ridge2f pretrained model

#* Forecast stock returns
#* @param ticker query integer Ticker to forecast (1 to n, where n is 
#* the number of columns in EuStockMarkets data from base R)
#* @param n_ahead query integer Forecast horizon (default: 14)
#* @param test_time_index date time index in test set (between 1 and 50)
#* @post /forecast
function(ticker = "DAX", 
         n_ahead = 14L, 
         test_time_index=1) {
  n_ahead <- as.integer(n_ahead)
  if (!(ticker %in% c("DAX", "SMI", "CAC", "FTSE"))) {
    return(list(error = "Invalid ticker"))
  }
  forecast_method(ticker, n_ahead, test_time_index)
}