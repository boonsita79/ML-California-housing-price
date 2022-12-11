### Class project 
### COM SCI-X 450.1 Introduction to Data Science
### Supervised Machine Learning

library(dplyr)
library(randomForest)

# ---------------------------------------------------------------
## 1. Access to dataset
# ---------------------------------------------------------------

## Import dataset and cast "ocean_proximity" to factor
CalHousing <- read.csv("housing.csv", header=TRUE) 
CalHousing$ocean_proximity <- as.factor(CalHousing$ocean_proximity)

#Display levels
levels(CalHousing$ocean_proximity)

# ---------------------------------------------------------------
## 2. EDA and Data Visualization
# ---------------------------------------------------------------

head(CalHousing)
tail(CalHousing)
summary(CalHousing)
#Commentary: 
#1) Found 207 missing values in total_bedrooms variables
#2) Medians of all variables are less than means implying right skewness
#3) Only 'ocean_proximity' is category variables 

#Perform a correlation analysis
round(cor(CalHousing[1:9]),digit=3)

#Commentary: There is a high correlation between households and total_rooms 
#also between households and population (potential Multicollinearity issue)

#Create histogram of each numeric variables
attach(CalHousing)
hist(longitude)
hist(latitude)
hist(housing_median_age)
hist(total_rooms)
hist(total_bedrooms)
hist(population)
hist(households)
hist(median_income)
hist(median_house_value)

#Produce boxplots of each numeric variables 
boxplot(longitude,main = "Longitude")
boxplot(latitude,main = "Latitude")
boxplot(housing_median_age,main = "Housing median age")
boxplot(total_rooms,main = "Total room")
boxplot(total_bedrooms,main = "Total bedroom")
boxplot(population,main = "Population")
boxplot(households,main = "Households")
boxplot(median_income,main = "Median income")
boxplot(median_house_value,main = "Median house value")

#Produce boxplots for the variables: housing_median_age, median_income,and median_house_value 
#"with respect" to the factor variable ocean_proximity.
boxplot(housing_median_age ~ ocean_proximity)
boxplot(median_income ~ ocean_proximity)
boxplot(median_house_value ~ ocean_proximity)

# ---------------------------------------------------------------
## 3. Data Transformation
# ---------------------------------------------------------------

#Replace Missing value with median
CalHousing$total_bedrooms[is.na(CalHousing$total_bedrooms)] <- median(CalHousing$total_bedrooms, na.rm=TRUE)
summary(CalHousing)

#Create new binary categorical variable from ocean_proximity
newcol <-(data.frame("INLAND"=(CalHousing$ocean_proximity=="INLAND")))
newcol$"ISLAND" <- CalHousing$ocean_proximity=="ISLAND"
newcol$"NEAR BAY" <- CalHousing$ocean_proximity=="NEAR BAY" 
newcol$"NEAR OCEAN" <- CalHousing$ocean_proximity=="NEAR OCEAN"
newcol$"<1H OCEAN"<- CalHousing$ocean_proximity=="<1H OCEAN"

newcol[] <- lapply(newcol, as.numeric)

#Add new columns to dataframe 
New_CalHousing <- cbind(CalHousing,newcol)

#Create new variables
New_CalHousing$mean_bedrooms <- (New_CalHousing$total_bedrooms/New_CalHousing$households)
New_CalHousing$mean_rooms <- (New_CalHousing$total_rooms/New_CalHousing$households)

#Remove 3 unused columns
New_CalHousing <- select(New_CalHousing, -c("total_bedrooms","total_rooms","ocean_proximity"))

#Rearrange columns in dataframe 
cleaned_housing <- New_CalHousing [, c("NEAR BAY", 
                                       "<1H OCEAN", 
                                       "INLAND",
                                       "NEAR OCEAN", 
                                       "ISLAND", 
                                       "longitude",
                                       "latitude", 
                                       "housing_median_age", 
                                       "population",
                                       "households", 
                                       "median_income", 
                                       "mean_bedrooms", 
                                       "mean_rooms", 
                                       "median_house_value")]

#Perform feature scaling each numeric variables
cleaned_housing[6:13] <- as.data.frame(scale(cleaned_housing[6:13]))

names(cleaned_housing)

# ---------------------------------------------------------------
## 4. Create Training and Test sets
# ---------------------------------------------------------------

# Split data set into training set and test set
n <- nrow(cleaned_housing)  # Number of observations = 20640
ntrain <- round(n*0.7)      # 70% for training set
set.seed(124)               # Set seed for reproducible results
tindex <- sample(n, ntrain) # Create an index

trainHouse <- cleaned_housing[tindex,]  # Create training set
testHouse <- cleaned_housing[-tindex,]  # Create test set

# ---------------------------------------------------------------
## 5. Supervised ML - Regression
# ---------------------------------------------------------------

#Split training set
train_x <-  trainHouse[1:13]    #Feature variables
train_y <-  as.vector(trainHouse$median_house_value) #Response variables

#Training model using Random forest 
rf = randomForest(x=train_x, y=train_y ,
                  ntree=500, importance=TRUE)
#See all metrics 
names(rf)

# ---------------------------------------------------------------
## 6. Evaluating Model performance
# ---------------------------------------------------------------

#Calculate RMSE (the square root of last rf$mse)
sqrt(tail(rf$mse,n=1))
# [1] 49165.43

#Split testing set
test_x <-  testHouse[1:13]  #Feature variables
test_y <-  as.vector(testHouse$median_house_value) #Response variables

# Calculate a vector of predicted median house value
prediction <- predict(rf, test_x)

# RMSE function
rmse <- function(y_hat, y)
{
  return(sqrt(mean((y_hat-y)^2)))
}

# Calculate RMSE for test set
rmse_test <- rmse(prediction, test_y)
rmse_test
# [1] 49786.27

varImpPlot(rf)

# ---------------------------------------------------------------
## Retraining model with 4 feature variables suggested by VarImplot
#Split training set
train_x2 <-  trainHouse[c("median_income",
                          "housing_median_age",
                          "longitude",
                          "latitude")] #New feature variables

train_y2 <-  as.vector(trainHouse$median_house_value) #Response variables

#Training model using Random forest 
rf2 = randomForest(x=train_x2, y=train_y2 ,
                  ntree=500, importance=TRUE)
#See all metrics 
names(rf2)

#Calculate RMSE (the square root of last rf$mse)
sqrt(tail(rf2$mse,n=1))
# [1] 50932.61

#Split testing set
test_x2 <-  testHouse[c("median_income",
                        "housing_median_age",
                        "longitude",
                        "latitude")] #New feature variables

test_y2 <-  as.vector(testHouse$median_house_value) #Response variables

# Calculate a vector of predicted median house value
prediction2 <- predict(rf2, test_x2)


# Calculate RMSE for test set
rmse_test2 <- rmse(prediction2, test_y2)
rmse_test2
# [1] 51083.02

# ---------------------------------------------------------------
