# label assignment and centrality index computation for friend and retweet networks

rm(list=ls())

wdir <- "D:/andre/Documenti/vaxxstance/"
setwd( wdir )



# functions assigning labels
source("functions.R")

EVAL_NETWORK=TRUE
library(readr)
library(igraph)
library(ggraph)
library(tm)
library(dplyr)
library(stringr)
library(tidytext)
library(ggplot2)
library(gridExtra)
library(tidyr)
library(widyr)
library(ggpubr)



# Spanish
# import train and test, assign labels
complete=make_user_label(c("D:/andre/Documenti/vaxxstance/csv/es_train/es_train.csv",
                       "D:/andre/Documenti/vaxxstance/es_test/es_test.csv"))


es_label <- complete

es_train <- read_csv("D:/andre/Documenti/vaxxstance/csv/es_train/es_train.csv", 
                     col_types = cols(user_id = col_character()))

# same function as above, focusing only on one dataset
source("f2.R")

# import and assign label
train=make_user_label_train("D:/andre/Documenti/vaxxstance/csv/es_train/es_train.csv")

es_label_train <- train

# import test set
es_user_test <- read_csv("D:/andre/Documenti/vaxxstance/es_test/es_user_test.csv",col_types = cols(user_id = col_character()))

# Spanish friends network
# import train and test sets
es_friend_train <- read_csv("D:/andre/Documenti/vaxxstance/csv/es_train/es_friend_train.csv",
                            col_types = cols(Source = col_character(),
                                             Target = col_character()))

es_friend_test <- read_csv("D:/andre/Documenti/vaxxstance/es_test/es_friend_test.csv",
                           col_types = cols(source = col_character(),
                                            target = col_character()))

es_friend_train$set <- "train"
es_friend_test$set <- "test"

names(es_friend_test)[1]="Source"
names(es_friend_test)[2]="Target"

es_friend <- rbind(es_friend_train,es_friend_test)

# convert edgelist as a graph
esfr <- as.matrix(es_friend[,c(1:2)])
g <- graph_from_edgelist(esfr, directed=TRUE)
V(g)$label <- V(g)$name

# centrality indices
V(g)$indegree <- degree(g, mode="in")           # INDegree centrality
V(g)$outdegree <- degree(g, mode="ou")          # OUTDegree centrality
V(g)$authorities <- authority.score(g)$vector   # "Authority" centrality
V(g)$closeness <- closeness(g)                  # Closeness centrality


centrality <- data.frame(row.names   = V(g)$name,
                         indegree    = V(g)$indegree,
                         outdegree   = V(g)$outdegree,
                         auth        = V(g)$authorities,
                         closeness   = V(g)$closeness)


centrality$user_id <- row.names(centrality)

# prepare datasets centrality_es_friends train and test
centrality_ES_train <- merge(es_label_train,centrality, by="user_id", all.x = TRUE)
centrality_ES_train <- centrality_ES_train[,c(1,4:7)]
centrality_ES_train[is.na(centrality_ES_train)] <- 0

centrality_ES_test <- merge(es_user_test,centrality, by="user_id", all.x = TRUE)
centrality_ES_test <- centrality_ES_test[,c(1,8:11)]
centrality_ES_test[is.na(centrality_ES_test)] <- 0



# Spanish retweet network
# import data
es_rt_train <- read_csv("D:/andre/Documenti/vaxxstance/csv/es_train/es_retweet_train.csv",
                        col_types = cols(Source = col_character(),
                                         Target = col_character()))

es_rt_test <- read_csv("D:/andre/Documenti/vaxxstance/es_test/es_retweet_test.csv",
                       col_types = cols(source = col_character(),
                                        target = col_character()))

names(es_rt_test)[1]="Source"
names(es_rt_test)[2]="Target"
names(es_rt_test)[3]="Weight"


es_rt_train$set <- "train"
es_rt_test$set <- "test"

es_rt <- rbind(es_rt_train[,c(1:4)],es_rt_test[,c(1:4)])

# converting into a graph object
esrt <- as.matrix(es_rt)
g <- graph_from_edgelist(esrt[,1:2], directed=TRUE)
V(g)$label <- V(g)$name
g <- set_edge_attr(g, "weight", value= es_rt$Weight)

# centrality indices
V(g)$indegree <- degree(g, mode="in")           # INDegree centrality
V(g)$outdegree <- degree(g, mode="ou")          # OUTdegree centrality
V(g)$hubs <- hub.score(g)$vector                # "Hub" centrality
V(g)$authorities <- authority.score(g)$vector   # "Authority" centrality

centrality <- data.frame(row.names   = V(g)$name,
                         indegree    = V(g)$indegree,
                         outdegree   = V(g)$outdegree,
                         hubs        = V(g)$hubs,
                         auth        = V(g)$authorities)

centrality$user_id <- row.names(centrality)

# prepare datasets
centrality_ES_train <- merge(es_label_train,centrality, by="user_id", all.x = TRUE)
centrality_ES_train <- centrality_ES_train[,c(1,4:7)]
centrality_ES_train[is.na(centrality_ES_train)] <- 0

centrality_ES_test <- merge(es_user_test,centrality, by="user_id", all.x = TRUE)
centrality_ES_test <- centrality_ES_test[,c(1,8:11)]
centrality_ES_test[is.na(centrality_ES_test)] <- 0


# basque
# see spanish for details

complete=make_user_label(c("D:/andre/Documenti/vaxxstance/csv/eu_train/eu_train.csv",
                       "D:/andre/Documenti/vaxxstance/eu_test/eu_test.csv"))




eu_label <- complete

eu_train <- read_csv("D:/andre/Documenti/vaxxstance/csv/eu_train/eu_train.csv", 
                     col_types = cols(user_id = col_character()))



train=make_user_label_train("D:/andre/Documenti/vaxxstance/csv/eu_train/eu_train.csv")

eu_label_train <- train

eu_user_test <- read_csv("D:/andre/Documenti/vaxxstance/eu_test/eu_user_test.csv",col_types = cols(user_id = col_character()))


# EU friends network

eu_friend_train <- read_csv("D:/andre/Documenti/vaxxstance/csv/eu_train/eu_friend_train.csv",
                            col_types = cols(Source = col_character(),
                                             Target = col_character()))

eu_friend_test <- read_csv("D:/andre/Documenti/vaxxstance/eu_test/eu_friend_test.csv",
                           col_types = cols(source = col_character(),
                                            target = col_character()))

eu_friend_train$set <- "train"
eu_friend_test$set <- "test"

names(eu_friend_test)[1]="Source"
names(eu_friend_test)[2]="Target"

eu_friend <- rbind(eu_friend_train,eu_friend_test)


esfr <- as.matrix(eu_friend[,c(1:2)])
g <- graph_from_edgelist(esfr, directed=TRUE)
V(g)$label <- V(g)$name


V(g)$indegree <- degree(g, mode="in")           # INDegree centrality
V(g)$eig <- evcent(g)$vector                    # Eigenvector centrality
V(g)$betweenness <- betweenness(g)              # Vertex betweenness centrality

centrality <- data.frame(row.names   = V(g)$name,
                         indegree    = V(g)$indegree,
                         # outdegree   = V(g)$outdegree,
                         # hubs        = V(g)$hubs,
                         # auth        = V(g)$authorities,
                         # closeness   = V(g)$closeness,
                         betweenness = V(g)$betweenness,
                         eigenvector = V(g)$eig)

centrality$user_id <- row.names(centrality)


centrality_EU_train <- merge(eu_label_train,centrality, by="user_id", all.x = TRUE)
centrality_EU_train <- centrality_EU_train[,c(1,4:6)]
centrality_EU_train[is.na(centrality_EU_train)] <- 0

centrality_EU_test <- merge(eu_user_test,centrality, by="user_id", all.x = TRUE)
centrality_EU_test <- centrality_EU_test[,c(1,8:10)]
centrality_EU_test[is.na(centrality_EU_test)] <- 0



# eu retweet
eu_rt_train <- read_csv("D:/andre/Documenti/vaxxstance/csv/eu_train/eu_retweet_train.csv",
                        col_types = cols(Source = col_character(),
                                         Target = col_character()))
eu_rt_timel_test <- read_csv("D:/andre/Documenti/vaxxstance/eu_test/eu_retweet_timeline_test.csv",
                             col_types = cols(Source = col_character(),
                                              Target = col_character()))
eu_user_test <- read_csv("D:/andre/Documenti/vaxxstance/eu_test/eu_user_test.csv", 
                         col_types = cols(user_id = col_character()))

eu_rt_test <- eu_rt_timel_test %>% filter(
  Target %in% eu_user_test$user_id
)


eu_rt_train$set <- "train"
eu_rt_test$set <- "test"

eu_rt <- rbind(eu_rt_train[,c(1:4)],eu_rt_test[,c(1:4)])


esrt <- as.matrix(eu_rt)
g <- graph_from_edgelist(esrt[,1:2], directed=TRUE)
V(g)$label <- V(g)$name
g <- set_edge_attr(g, "weight", value= eu_rt$Weight)


V(g)$outdegree <- degree(g, mode="ou")          # OUTdegree centrality
V(g)$eig <- evcent(g)$vector                    # Eigenvector centrality
V(g)$authorities <- authority.score(g)$vector   # "Authority" centrality
V(g)$closeness <- closeness(g)                  # Closeness centrality

centrality <- data.frame(row.names   = V(g)$name,
                         outdegree   = V(g)$outdegree,
                         auth        = V(g)$authorities,
                         closeness   = V(g)$closeness,
                         eigenvector = V(g)$eig)

centrality$user_id <- row.names(centrality)


centrality_EU_train <- merge(eu_label_train,centrality, by="user_id", all.x = TRUE)
centrality_EU_train <- centrality_EU_train[,c(1,4:7)]
centrality_EU_train[is.na(centrality_EU_train)] <- 0

centrality_Eu_test <- merge(eu_user_test,centrality, by="user_id", all.x = TRUE)
centrality_Eu_test <- centrality_Eu_test[,c(1,8:11)]
centrality_Eu_test[is.na(centrality_Eu_test)] <- 0


