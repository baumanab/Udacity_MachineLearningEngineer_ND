
# imports
library(ggplot2)
library(tidyr)
library(dplyr)
library(data.table)
library(RColorBrewer)
library(scales)
library(gridExtra)
source('plotting_tools.R')

# read training data
train <- read.csv('train.csv') # train data merged with weather data (both stations)
train_station_1 <- read.csv('train_station_1.csv') # station 1 weather only
train_station_2 <- read.csv('train_station_2.csv') # station 2 weather only

# convert staton to factor for use in plotting
train$Station <- factor(train$Station)

glimpse(train)



#distribution of temperatures via overlaid density plots
ggplot(train, aes(x=Tmax, fill=Station)) +
  geom_density(alpha= 0.2) +
  geom_density(aes(x = blended_Tmax), alpha= 0, linetype= 'dashed') +
  my_theme() +
  ggtitle("Density Plot of Tmax by Station")

ggplot(train, aes(x=Tmin, fill=Station)) +
  geom_density(alpha= 0.2) +
  geom_density(aes(x = blended_Tmin), alpha= 0, linetype= 'dashed') +
  my_theme() +
  ggtitle("Density Plot of Tmin by Station")

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
  scale_x_log10() +
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
  coord_flip() +
  ggtitle("WNV Rate by Block") +
  xlab("Block") +
  ylab("WNV Rate")

ggplot(train_station_1, aes(x=reorder(Street, WnvPresent, function(x){mean(x)}),
                            y= WnvPresent)) +
  geom_bar(stat= "summary", fun.y= "mean", color= "blue", width= 0.05) +
  coord_flip() +
  ggtitle("WNV Rate by Street") +
  xlab("Street") +
  ylab("WNV Rate")

ggplot(train_station_1, aes(x=reorder(Species, WnvPresent, function(x){mean(x)}),
                            y= WnvPresent)) +
  geom_bar(stat= "summary", fun.y= "mean", color= "blue",
           width= 0.1) +
  coord_flip() +
  ggtitle("WNV Rate by Species") +
  xlab("Species") +
  ylab("WNV Rate")

ggplot(train_station_1, aes(x=reorder(month, WnvPresent, function(x){mean(x)}),
                            y= WnvPresent)) +
  geom_bar(stat= "summary", fun.y= "mean", color= "blue",
           width= 0.1) +
  coord_flip()

ggplot(train_station_1, aes(x=reorder(week, WnvPresent, function(x){mean(x)}),
                            y= WnvPresent)) +
  geom_bar(stat= "summary", fun.y= "mean", color= "blue",
           width= 0.1) +
  coord_flip() +
  ggtitle("WNV Rate by Week of Year") +
  xlab("Week of Year") +
  ylab("WNV Rate")

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

ggplot(train, aes(x=reorder(DayLength_NearH, WnvPresent, function(x){mean(x)}),
                            y= WnvPresent)) +
  geom_bar(stat= "summary", fun.y= "mean", color= "blue",
           width= 0.1, position= "dodge") +
  coord_flip() +
  ggtitle("WNV Rate by Day Length") +
  xlab("Length of Day (hours)") +
  ylab("WNV Rate")

ggplot(train, aes(x=reorder(NightLength_NearH, WnvPresent, function(x){mean(x)}),
                  y= WnvPresent, fill= Station)) +
  geom_bar(stat= "summary", fun.y= "mean", color= "blue",
           width= 0.1, position= "dodge") +
  coord_flip()





# workbench https://www.r-bloggers.com/from-continuous-to-categorical/


ggplot(train, aes(x= cut(train$Tavg, seq(40,100,5), right=FALSE),y= WnvPresent,
                  color= Station)) +
  geom_point(stat= "summary", fun.y= 'mean', size= 2) +
  ggtitle("WNV Rate by Average Temperature") +
  xlab("Celsius") +
  ylab("WNV Rate")
  

ggplot(train, aes(x= cut(train$Tmin, seq(40,100,5), right=FALSE),y= WnvPresent,
                  color= Station)) +
  geom_point(stat= "summary", fun.y= 'mean', size= 2) +
  ggtitle("WNV Rate by Minimum Temperature") +
  xlab("Celsius") +
  ylab("WNV Rate")

ggplot(train, aes(x= cut(train$blended_Tmin, seq(40,100,5), right=FALSE),y= WnvPresent,
                  color= Station)) +
  geom_point(stat= "summary", fun.y= 'mean', size= 2) 
  ggtitle("WNV Rate by Minimum Temperature") +
  xlab("Celsius") +
  ylab("WNV Rate")

ggplot(train, aes(x= cut(train$Tmax, seq(40,100,5), right=FALSE),y= WnvPresent,
                  color= Station)) +
  geom_point(stat= "summary", fun.y= 'mean', size= 2) +
  ggtitle("WNV Rate by Maximum Temperature") +
  xlab("Celsius") +
  ylab("WNV Rate")

ggplot(train, aes(x= cut(train$blended_Tmax, seq(40,100,5), right=FALSE),y= WnvPresent,
                  color= Station)) +
  geom_point(stat= "summary", fun.y= 'mean', size= 2)


ggplot(train, aes(x= cut(train$Depart, seq(-12,18,5), right=FALSE),y= WnvPresent)) +
  geom_point(stat= "summary", fun.y= 'mean', size= 2) +
  ggtitle("WNV Rate by Depart") +
  xlab("Celsius") +
  ylab("WNV Rate")

ggplot(train, aes(x= cut(train$Heat, seq(0,15,3), right=FALSE),y= WnvPresent)) +
  geom_point(stat= "summary", fun.y= 'mean', size= 2) +
  ggtitle("WNV Rate by Heat") +
  xlab("Celsius") +
  ylab("WNV Rate")

ggplot(train, aes(x= cut(train$Cool, seq(0,22,4), right=FALSE),y= WnvPresent)) +
  geom_point(stat= "summary", fun.y= 'mean', size= 2) +
  ggtitle("WNV Rate by Cool") +
  xlab("Celsius") +
  ylab("WNV Rate")


ggplot(train, aes(x= cut(train$PrecipTotal, seq(0,4,1), right=FALSE),y= WnvPresent)) +
  geom_point(stat= "summary", fun.y= 'mean', size= 2) +
  ggtitle("WNV Rate by Total Precipitation") +
  xlab("Precipitation (inches)") +
  ylab("WNV Rate")


# sunrise, sunset

ggplot(train, aes(x= cut(train$Sunrise_hours, seq(4,6,0.1), right=FALSE),y= WnvPresent)) +
  geom_point(stat= "summary", fun.y= 'mean', size= 2) +
  ggtitle("WNV Rate by Sunrise Hours") +
  xlab("Hours since Midnight") +
  ylab("WNV Rate")

ggplot(train, aes(x= cut(train$Sunset_hours, seq(17,20,0.25), right=FALSE),y= WnvPresent)) +
  geom_point(stat= "summary", fun.y= 'mean', size= 2)

# day of year exploration

ggplot(train, aes(x=day_of_year)) +
  geom_histogram() +
  ggtitle("Histogram of Day of Year")

ggplot(train_station_1, aes(x=reorder(day_of_year, WnvPresent, function(x){mean(x)}),
                            y= WnvPresent)) +
  geom_line(aes(group=1),stat= "summary", fun.y= "mean", color= "blue") +
  coord_flip() +
  ggtitle("WNV Rate by Day of Year") +
  xlab("Day of Year") +
  ylab("WNV Rate")

ggplot(train_station_1, aes(x=reorder(day_of_year, WnvPresent, function(x){mean(x)}),
                            y= WnvPresent)) +
  geom_line(aes(group=1),stat= "summary", fun.y= "mean", color= "blue") +
  ggtitle("WNV Rate by Day of Year") +
  xlab("Day of Year") +
  ylab("WNV Rate") +
  theme(axis.text.x  = element_text(angle=90, vjust=0.5))

a= ggplot(train_station_1, aes(x=week, WnvPresent, function(x){mean(x)},
                               y= WnvPresent))+
  geom_line(aes(group=1),stat= "summary", fun.y= "mean", color= "green", size= 2) +
  ylab("WNV Rate") +
  theme(axis.text.x  = element_blank(), axis.title.x=element_blank()) +
  ggtitle("Grid of Attributes by Week of Year")


b= ggplot(train_station_1, aes(x= week, y= Tmax )) +
   geom_point(color= 'tan') +
  theme(axis.text.x  = element_blank(), axis.title.x=element_blank())+
  ylab("Tmax(F)")

c= ggplot(train_station_1, aes(x= week, y= PrecipTotal )) +
  geom_point(color= "blue") +
  theme(axis.text.x  = element_blank(), axis.title.x=element_blank())+
  ylab("Precipitation(inches)")


d= ggplot(train_station_1, aes(x= week, y= Tmin )) +
   geom_point(color= 'orange') +
  theme(axis.text.x  = element_blank(), axis.title.x=element_blank()) +
  ylab("Tmin(F)")


e= ggplot(train_station_1, aes(x= week, y= Tavg)) +
   geom_point(color= 'red') +
   xlab("Week of Year") +
   ylab("Tmin(F)")


grid.arrange(a,c, b, d,e, nrow=5)


ggplot(train_station_1, aes(x=day, WnvPresent, function(x){mean(x)},
                            y= WnvPresent)) +
  geom_line(aes(group=1),stat= "summary", fun.y= "mean", color= "blue") +
  ggtitle("WNV Rate by Day of Months") +
  xlab("Day of Month") +
  ylab("WNV Rate")
