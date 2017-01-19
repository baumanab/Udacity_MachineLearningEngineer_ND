
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

glimpse(train)



#distribution of temperatures via overlaid histograms
ggplot(train, aes(x=Tmax, fill=Station)) +
  geom_density(alpha= 0.2) +
  geom_density(aes(x = blended_Tmax), alpha= 0, linetype= 'dashed') +
  my_theme()

ggplot(train, aes(x=Tmin, fill=Station)) +
  geom_density(alpha= 0.2) +
  geom_density(aes(x = blended_Tmin), alpha= 0, linetype= 'dashed') +
  my_theme()

ggplot(train, aes(x=Tavg, fill=Station)) +
  geom_density(alpha= 0.2) +
  my_theme()

ggplot(train, aes(x=PrecipTotal, fill=Station)) +
  geom_density(alpha= 0.2) +
  geom_density(aes(x = blended_PrecipTotal), alpha= 0, linetype= 'dashed') +
  my_theme()

ggplot(train, aes(x=PrecipTotal, fill=Station)) +
  geom_density(alpha= 0.2) +
  geom_density(aes(x = blended_PrecipTotal), alpha= 0, linetype= 'dashed') +
  scale_x_log10() +
  my_theme()

ggplot(train, aes(x=Depart, fill=Station)) + # station 2 forward filled from station 1
  geom_density(alpha= 0.2) +
  geom_density(aes(x = blended_Depart), alpha= 0, linetype= 'dashed') +
  my_theme()

ggplot(train, aes(x=Heat, fill=Station)) +
  geom_density(alpha= 0.2) +
  geom_density(aes(x = blended_Heat), alpha= 0, linetype= 'dashed') +
  my_theme()

ggplot(train, aes(x=Heat, fill=Station)) +
  geom_density(alpha= 0.2) +
  geom_density(aes(x = blended_Tmax), alpha= 0, linetype= 'dashed') +
  my_theme()

ggplot(train, aes(x=Heat, fill=Station)) +
  geom_density(alpha= 0.2) +
  geom_density(aes(x = blended_Tmax), alpha= 0, linetype= 'dashed') +
  scale_x_log10()
  my_theme()

ggplot(train, aes(x=Cool, fill=Station)) +
  geom_density(alpha= 0.2) +
  geom_density(aes(x = blended_Cool), alpha= 0, linetype= 'dashed') +
  my_theme()



# bar plots

ggplot(train_station_1, aes(x=reorder(Block, WnvPresent, function(x){mean(x)}),
                            y= WnvPresent)) +
  geom_bar(stat= "summary", fun.y= "mean", color= "blue",
           width= 0.05) +
  coord_flip()

ggplot(train_station_1, aes(x=reorder(Street, WnvPresent, function(x){mean(x)}),
                            y= WnvPresent)) +
  geom_bar(stat= "summary", fun.y= "mean", color= "blue", width= 0.05) +
  coord_flip()

ggplot(train_station_1, aes(x=reorder(Species, WnvPresent, function(x){mean(x)}),
                            y= WnvPresent)) +
  geom_bar(stat= "summary", fun.y= "mean", color= "blue",
           width= 0.1) +
  coord_flip()

ggplot(train_station_1, aes(x=reorder(month, WnvPresent, function(x){mean(x)}),
                            y= WnvPresent)) +
  geom_bar(stat= "summary", fun.y= "mean", color= "blue",
           width= 0.1) +
  coord_flip()

ggplot(train_station_1, aes(x=reorder(week, WnvPresent, function(x){mean(x)}),
                            y= WnvPresent)) +
  geom_bar(stat= "summary", fun.y= "mean", color= "blue",
           width= 0.1) +
  coord_flip()

ggplot(train_station_1, aes(x=reorder(day, WnvPresent, function(x){mean(x)}),
                            y= WnvPresent)) +
  geom_bar(stat= "summary", fun.y= "mean", color= "blue",
           width= 0.1) +
  coord_flip()

ggplot(train_station_1, aes(x=reorder(day_of_year, WnvPresent, function(x){mean(x)}),
                            y= WnvPresent)) +
  geom_bar(stat= "summary", fun.y= "mean", color= "blue",
           width= 0.1) +
  coord_flip()

ggplot(train, aes(x=reorder(DayLength, WnvPresent, function(x){mean(x)}),
                            y= WnvPresent, fill= Station)) +
  geom_bar(stat= "summary", fun.y= "mean", color= "blue",
           width= 0.1, position= "dodge") +
  coord_flip()




# workbench https://www.r-bloggers.com/from-continuous-to-categorical/


ggplot(train, aes(x= cut(train$Tavg, seq(40,100,5), right=FALSE),y= WnvPresent,
                  color= Station)) +
  geom_point(stat= "summary", fun.y= 'mean', size= 2)

ggplot(train, aes(x= cut(train$Tmin, seq(40,100,5), right=FALSE),y= WnvPresent,
                  color= Station)) +
  geom_point(stat= "summary", fun.y= 'mean', size= 2)

ggplot(train, aes(x= cut(train$blended_Tmin, seq(40,100,5), right=FALSE),y= WnvPresent,
                  color= Station)) +
  geom_point(stat= "summary", fun.y= 'mean', size= 2)

ggplot(train, aes(x= cut(train$Tmax, seq(40,100,5), right=FALSE),y= WnvPresent,
                  color= Station)) +
  geom_point(stat= "summary", fun.y= 'mean', size= 2)

ggplot(train, aes(x= cut(train$blended_Tmax, seq(40,100,5), right=FALSE),y= WnvPresent,
                  color= Station)) +
  geom_point(stat= "summary", fun.y= 'mean', size= 2)


ggplot(train, aes(x= cut(train$Depart, seq(-12,18,5), right=FALSE),y= WnvPresent)) +
  geom_point(stat= "summary", fun.y= 'mean', size= 2)

ggplot(train, aes(x= cut(train$Heat, seq(0,15,3), right=FALSE),y= WnvPresent)) +
  geom_point(stat= "summary", fun.y= 'mean', size= 2)

ggplot(train, aes(x= cut(train$Cool, seq(0,22,4), right=FALSE),y= WnvPresent)) +
  geom_point(stat= "summary", fun.y= 'mean', size= 2)


ggplot(train, aes(x= cut(train$PrecipTotal, seq(0,4,1), right=FALSE),y= WnvPresent)) +
  geom_point(stat= "summary", fun.y= 'mean', size= 2)


# sunrise, sunset

train$Sunrise



