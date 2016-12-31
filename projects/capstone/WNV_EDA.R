
# imports
library(ggplot2)
library(tidyr)
library(dplyr)
library(data.table)
library(RColorBrewer)
library(scales)
source('plotting_tools.R')

# read training data
train <- read.csv('train.csv') # train data merged with weather data (both stations)
train_station_1 <- read.csv('train_station_1.csv') # station 1 weather only
train_station_2 <- read.csv('train_station_2.csv') # station 2 weather only

# convert staton to factor for use in plotting
train$Station <- factor(train$Station)

#cglimpse(train)



#distribution of temperatures via overlaid histograms
ggplot(train, aes(x=Tmax, fill=Station)) +
  geom_density(alpha = 0.3) +
  my_theme()

ggplot(train, aes(x=Tmin, fill=Station)) +
  geom_density(alpha = 0.3) +
  my_theme()

ggplot(train, aes(x=Tavg, fill=Station)) +
  geom_density(alpha = 0.3) +
  my_theme()




