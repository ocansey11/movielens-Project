##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(janitor)) install.packages("janitor", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")



library(tidyverse)
library(caret)
library(data.table)
library(janitor)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                            title = as.character(title),
#                                            genres = as.character(genres))

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)




# SUMMARY OF EDX BEFORE CLEANING
head(edx)
dim(edx)
edx %>%
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

unique(edx$genres)
# i am thinking of removing data that have no genre listed. since they are only 7 
count(edx %>% filter(genres %in% "(no genres listed)"))

edx %>% filter(movieId == 8606)
# You'll notice they are all from the same movie 'Pull My daisy'

# Cleaned edx dataset
edx <- edx %>% filter(!movieId == 8606)


# OVERVIEW ANALYSES WITH CROSS TAB
# This is to help me skim through the data before i start my analyses. It also helps me plan how to build charts and graphs
tabyl(edx, genres) %>% arrange(desc(n))
tabyl(edx, rating) %>% arrange(desc(n))
tabyl(edx, title) %>% arrange(desc(n))



# VISUALIZATION
# Lets look at the most Popular Movies
edx %>%
  group_by(title) %>%
  summarize(count = n()) %>%
  arrange(-count) %>%
  top_n(15, count) %>%
  ggplot(aes(count, reorder(title, count))) +
  geom_bar(color = "black", fill= '#00abff',  stat = "identity") +
  xlab("Count") +
  ylab("Movies")


# Now lets look at the number of ratings spread across our users
edx %>% group_by(userId) %>%
  summarize(count = n()) %>%
  ggplot(aes(count)) +
  geom_histogram(color = "black", fill = "#00abff", bins = 40) +
  xlab("Ratings") +
  ylab("Users") +
  scale_x_log10() 


count(tabyl(edx, userId) %>% arrange(n) %>% filter(n < 30 ))
#notice we have 15k users who haven't rated at least 30 movies

count(tabyl(edx, userId) %>% arrange(n) %>% filter(n < 11 ))
# However there's only one user that has watched fewer than 11 movies in our data set. that seems sufficient to carry out the study.



# DIVIDING EDX IN TRAINING AND TEST DDATA
index_edx <-  createDataPartition(y = edx$rating, times = 1, p = 0.8, list =  FALSE) 
train_set <- edx[index_edx,]
test_set <- edx[-index_edx,]


test_set <- test_set %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")



# a FUNCTION THAT WILL CALCULATE RMSE 
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}








# MACHINE LEARNING
# Following the idea from the course book we will build on a MODEL idea slowly till we get an rmse loWer than 0.8

#1. Simple model
# We use the avg of the ratings to build a simple predictive model.
mu_ratings <- mean(train_set$rating)
mu_ratings

# Our simple model looks like this mathematically  =>  Yu,i = μ_hat +  εu,i
# where Y represents our predicted rating, u_hat our avg rating for all movies and epsolon represnts any minor error 

simple_model_rmse <- RMSE(test_set$rating, mu_ratings)

#  Meanwhile i would create a table to keep track of all the rmse results as we improve the model
rmse_results <- tibble(method = "Simple Model using Average", RMSE = simple_model_rmse)





# 2. Movie effect model + simple model.  to measure their bias from the avg movie ratings we got from mu_ratings 
mu <- mean(train_set$rating)
movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))


movie_avgs %>% ggplot(aes(b_i)) +
  geom_histogram(color = "black", fill = "#00abff", bins = 10) +
  xlab("Movie Bias") +
  ylab("Count") 



# Now adding b_i into our first simple model gives us something like this  yu,i = μ_hat + b_hati
predicted_ratings <- mu + test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
movie_effect <- RMSE(predicted_ratings, test_set$rating)


rmse_results <- add_row(.data = rmse_results, method= 'Simple Model + Movie effects', RMSE =  movie_effect )


# 3.User Effect

train_set %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating)) %>%
  filter(n()>=100) %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 30, fill = "#00abff")


# User Effect Model
user_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
user_effect <- RMSE(predicted_ratings, test_set$rating)


rmse_results <- add_row(.data = rmse_results, method= 'Simple Model + Movie Effect + User effects', RMSE = user_effect )

# Now We  need to get an Rmse lower than 0.86



# Regularization
# First off we need to make sure that movies with little as 100 ratings have the same weight as movies with 1000 ratings

lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(x){
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+x))
  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+x))
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})

lambda_rmse <-  lambdas[which.min(rmses)]
qplot(lambdas, rmses)
# lambda of 4.75 provides the lowest RMSE error

rmse_results <- add_row(.data = rmse_results, method= 'Simple Model + Movie Effect + User effects + Regularization', RMSE = min(rmses))






# Matrix Factorization
# https://www.r-bloggers.com/2016/07/recosystem-recommender-system-using-parallel-matrix-factorization/
# Above is a link to better understand how Recosystem works

# This will take a while
library(recosystem)
set.seed(1, sample.kind="Rounding")
train_reco <- with(train_set, data_memory(user_index = userId, item_index = movieId, rating = rating))
test_reco <- with(test_set, data_memory(user_index = userId, item_index = movieId, rating = rating))
r <- Reco()

tuned_reco <- r$tune(train_reco, opts = list(dim = c(20, 30),
                                            costp_l2 = c(0.01, 0.1),
                                            costq_l2 = c(0.01, 0.1),
                                            lrate = c(0.01, 0.1),
                                            nthread = 4,
                                            niter = 10))

r$train(train_reco, opts = c(tuned_reco$min, nthread = 4, niter = 30))
results_reco <- r$predict(test_reco, out_memory())


factorization_rmse <- RMSE(results_reco, test_set$rating)

rmse_results <- add_row(.data = rmse_results, method= 'Factorization using Reco library', RMSE = factorization_rmse)
# our Rmse now is below 0.86 at 0.79



# Validation with our VALIDATION data set, with edx as our training data set
set.seed(1, sample.kind="Rounding")
edx_reco <- with(edx, data_memory(user_index = userId, item_index = movieId, rating = rating))
validation_reco <- with(validation, data_memory(user_index = userId, item_index = movieId, rating = rating))
r <- Reco()

tuned_reco <- r$tune(edx_reco, opts = list(dim = c(20, 30),
                                          costp_l2 = c(0.01, 0.1),
                                          costq_l2 = c(0.01, 0.1),
                                          lrate = c(0.01, 0.1),
                                          nthread = 4,
                                          niter = 10))

r$train(edx_reco, opts = c(tuned_reco$min, nthread = 4, niter = 30))

final_reco <- r$predict(validation_reco, out_memory())
final_rmse <-  RMSE(final_reco,validation$rating)
rmse_results <- add_row(.data = rmse_results, method= 'Final Rmse', RMSE = final_rmse)

