library(tidyverse)
library(tidymodels)
library(vroom)
library(prophet)
library(modeltime)
library(timetk)

# Load the training and testing datasets
train_data <- vroom('train.csv') 
test_data <- vroom('test.csv')   

# Convert the 'date' column to Date format
train_data$date <- as.Date(train_data$date)
test_data$date <- as.Date(test_data$date)

# Determine the number of unique stores and items in the training data
total_stores <- max(train_data$store)
total_items <- max(train_data$item)

# Loop through each store-item combination to train models and make predictions
for (store_num in 1:total_stores) {
  for (item_num in 1:total_items) {
    
    # Subset the training and testing data for the current store-item combination
    train_filtered <- train_data %>% filter(store == store_num, item == item_num)
    test_filtered <- test_data %>% filter(store == store_num, item == item_num)
    
    # Perform a time-series split for cross-validation
    ts_cv_split <- time_series_split(train_filtered, assess = "3 months", cumulative = TRUE)
    
    # Define and fit a Prophet model
    prophet_model <- prophet_reg() %>%
      set_engine(engine = "prophet") %>%
      fit(sales ~ date, data = training(ts_cv_split))
    
    # Calibrate the model using the testing portion of the split
    calibrated_model <- modeltime_calibrate(prophet_model, new_data = testing(ts_cv_split))
    
    # Refit the model to the entire training dataset
    refitted_model <- calibrated_model %>% modeltime_refit(data = train_filtered)
    
    # Generate sales forecasts for the test dataset
    forecast_results <- refitted_model %>%
      modeltime_forecast(new_data = test_filtered, actual_data = train_filtered) %>%
      filter(!is.na(.model_id)) %>%
      mutate(id = test_filtered$id) %>%
      select(id, .value) %>%
      rename(sales = .value)
    
    # Combine the forecasts for all store-item pairs
    if (store_num == 1 && item_num == 1) {
      all_forecasts <- forecast_results
    } else {
      all_forecasts <- bind_rows(all_forecasts, forecast_results)
    }
  }
}

# Save the combined forecasts to a CSV file
vroom_write(all_forecasts, file = "submission.csv", delim = ",")
