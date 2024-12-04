library(tidymodels)
library(vroom)
library(forecast)
library(patchwork)

# Read in data
item_test <- vroom("test.csv")
item_train <- vroom("train.csv")

# Filter down to just 1 store item for exploration and model building
store_item <- item_train |> 
  filter(store == 1, item == 1)

# Create time series plot
tsplot <- store_item |> 
  ggplot(mapping = aes(x = date, y = sales)) +
  geom_line() +
  geom_smooth(se = FALSE)

# Create autocorrelation function plots
acf1mo <- store_item |> 
  pull(sales) |> 
  forecast::ggAcf()

acf2yr <- store_item |> 
  pull(sales) |> 
  forecast::ggAcf(lag.max = 2*365)

# Another store item
store_item2 <- item_train |> 
  filter(store == 5, item == 11)

tsplot2 <- store_item2 |> 
  ggplot(mapping = aes(x = date, y = sales)) +
  geom_line() +
  geom_smooth(se = FALSE)

acf1mo2 <- store_item2 |> 
  pull(sales) |> 
  forecast::ggAcf()

acf2yr2 <- store_item2 |> 
  pull(sales) |> 
  forecast::ggAcf(lag.max = 2*365)

# Another store item
store_item3 <- item_train |> 
  filter(store == 6, item == 25)

tsplot3 <- store_item3 |> 
  ggplot(mapping = aes(x = date, y = sales)) +
  geom_line() +
  geom_smooth(se = FALSE)

acf1mo3 <- store_item3 |> 
  pull(sales) |> 
  forecast::ggAcf()

acf2yr3 <- store_item3 |> 
  pull(sales) |> 
  forecast::ggAcf(lag.max = 2*365)

# Create 2x3 panel plot
panel_plot <- ((tsplot2|acf1mo2|acf2yr2)/
                 (tsplot3|acf1mo3|acf2yr3))


library(vroom)
library(tidymodels)
library(tidyverse)
library(forecast)
library(yardstick)
library(modeltime)
library(timetk)

store_train <- vroom('train.csv')
store_test <- vroom('test.csv')  

storeItemTrain <- store_train %>%
  filter(store==8, item==11)

storeItemTest <- store_test %>%
  filter(store==8, item==11)

# train <- store_train %>% filter(store==6,item==13)
cv_split <- time_series_split(storeItemTrain,assess = "3 months", cumulative = TRUE)
cv_split %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)

arima_recipe <- recipe(sales ~ ., data = storeItemTrain) %>% # linear model part
  step_date(date, features=c("month", "dow", "doy", "year")) %>%
  step_mutate(date_month = as.numeric(date_month)) %>%
  step_mutate(date_dow = as.numeric(date_dow)) %>%
  step_range(date_month, min=0,max=pi) %>%
  step_mutate(sinMonth = sin(date_month), cosMonth = cos(date_month)) %>%
  step_range(date_dow, min=0, max=pi) %>%
  step_mutate(sinDOW = sin(date_dow), cosMonth = cos(date_dow))



arima_model <- arima_reg(seasonal_period=365,
                         non_seasonal_ar=5, #default max p to tune
                         non_seasonal_ma=5, #default max q to tune
                         seasonal_ar=2, #default max P to tune
                         seasonal_ma=2, #default max Q to tune
                         non_seasonal_differences = 2, #default max d to tune
                         seasonal_differences = 2 #default mox D to tune
) %>%
  set_engine("auto_arima")

arima_wf <- workflow() %>%
  add_recipe(arima_recipe) %>%
  add_model(arima_model) %>%
  fit(data = training(cv_split))

##Calibrate (tune) the models (find p,d,q,P,D,Q)
cv_results <- modeltime_calibrate(arima_wf,
                                  new_data = testing(cv_split))


## Visualize results
p3 = cv_results %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = training(cv_split),
  ) %>%
  plot_modeltime_forecast(.interactive=FALSE)


## Now that you have calibrated (tuned), refit to whole dataset
fullfit <- cv_results %>%
  modeltime_refit(data = storeItemTrain)


## Predict for all the observations in storeItemTest
p4 = fullfit %>%
  modeltime_forecast(
    new_data = storeItemTest,
    actual_data = storeItemTrain
  ) %>%
  plot_modeltime_forecast(.interactive = FALSE)

p1 = plot_modeltime_forecast()

plotly::subplot(p3,p4,nrows = 2)

library(gridExtra)
grid.arrange(p3,p4, ncol = 2)



library(modeltime)
library(timetk)
library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(workflows)
library(forecast)
library(patchwork)
library(ranger)
library(plotly)
library(prophet)

# Load datasets
train_data <- vroom("./train.csv")
test_data <- vroom("./test.csv")

# Filter data for Store 5 and Item 25
train_subset <- train_data %>% filter(store == 5, item == 25)
test_subset <- test_data %>% filter(store == 5, item == 25)

# Perform time-series split
cv_data <- time_series_split(train_subset, assess = "3 months", cumulative = TRUE)
cv_data %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, sales, .interactive = FALSE)

# Fit Prophet model
model_prophet <- prophet_reg() %>%
  set_engine("prophet") %>%
  fit(sales ~ date, data = training(cv_data))

# Calibrate model and assess accuracy
calibration_results <- modeltime_calibrate(model_prophet, new_data = testing(cv_data))

calibration_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = FALSE)

# Plot cross-validation predictions
cv_plot1 <- calibration_results %>%
  modeltime_forecast(new_data = testing(cv_data), actual_data = training(cv_data)) %>%
  plot_modeltime_forecast(.interactive = FALSE, .title = "CV Predictions: Store 5, Item 25")

# Refit the model and forecast
refitted_model <- calibration_results %>%
  modeltime_refit(data = train_subset)

forecast_plot1 <- refitted_model %>%
  modeltime_forecast(new_data = test_subset, actual_data = train_subset) %>%
  plot_modeltime_forecast(.interactive = FALSE, .title = "3-Month Forecast: Store 5, Item 25")

# Repeat process for Store 2, Item 15
train_subset <- train_data %>% filter(store == 2, item == 15)
test_subset <- test_data %>% filter(store == 2, item == 15)

cv_data <- time_series_split(train_subset, assess = "3 months", cumulative = TRUE)
cv_data %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, sales, .interactive = FALSE)

model_prophet <- prophet_reg() %>%
  set_engine("prophet") %>%
  fit(sales ~ date, data = training(cv_data))

calibration_results <- modeltime_calibrate(model_prophet, new_data = testing(cv_data))

calibration_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = FALSE)

cv_plot2 <- calibration_results %>%
  modeltime_forecast(new_data = testing(cv_data), actual_data = training(cv_data)) %>%
  plot_modeltime_forecast(.interactive = FALSE, .title = "CV Predictions: Store 2, Item 15")

refitted_model <- calibration_results %>%
  modeltime_refit(data = train_subset)

forecast_plot2 <- refitted_model %>%
  modeltime_forecast(new_data = test_subset, actual_data = train_subset) %>%
  plot_modeltime_forecast(.interactive = FALSE, .title = "3-Month Forecast: Store 2, Item 15")

# Combine all plots in a grid layout
plotly::subplot(cv_plot1, cv_plot2, forecast_plot1, forecast_plot2, nrows = 2, margin = 0.1) %>%
  layout(
    showlegend = FALSE,
    title = "",
    annotations = list(
      list(
        x = 0.2,
        y = 1.0,
        text = "CV Predictions: Store 5, Item 25",
        xref = "paper",
        yref = "paper",
        xanchor = "center",
        yanchor = "bottom",
        showarrow = FALSE
      ),
      list(
        x = 0.8,
        y = 1.0,
        text = "CV Predictions: Store 2, Item 15",
        xref = "paper",
        yref = "paper",
        xanchor = "center",
        yanchor = "bottom",
        showarrow = FALSE
      ),
      list(
        x = 0.2,
        y = 0.4,
        text = "3-Month Forecast: Store 5, Item 25",
        xref = "paper",
        yref = "paper",
        xanchor = "center",
        yanchor = "bottom",
        showarrow = FALSE
      ),
      list(
        x = 0.8,
        y = 0.4,
        text = "3-Month Forecast: Store 2, Item 15",
        xref = "paper",
        yref = "paper",
        xanchor = "center",
        yanchor = "bottom",
        showarrow = FALSE
      )
    )
  )
