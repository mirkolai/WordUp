# Centrality indices - augmented network - retweets

rm(list=ls())

wdir <- "D:/andre/Documenti/vaxxstance/"
setwd( wdir )

library(readr)
library(igraph)

# Basque

# import edgelist
edeges_eu <- read_csv("edeges_eu.csv", col_types = cols(source = col_character(), 
                                                        target = col_character()))

# load labelled data and test set
load("D:/andre/Documenti/vaxxstance/rifatti/EU_label.Rdata")
load("D:/andre/Documenti/vaxxstance/rifatti/EU_label_train.Rdata")
load("D:/andre/Documenti/vaxxstance/rifatti/EU_user_test.Rdata")

eu_label <- dati

eu_label_train <- dato

names(edeges_eu)[1]="Source"
names(edeges_eu)[2]="Target"

# convert edgelist into a graph object
esrt <- as.matrix(edeges_eu)
g <- graph_from_edgelist(esrt[,1:2], directed=TRUE)
V(g)$label <- V(g)$name

# centrality indices
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

# train and test sets features
centrality_EU_train <- merge(eu_label_train,centrality, by="user_id", all.x = TRUE)
centrality_EU_train <- centrality_EU_train[,c(1,4:7)]
centrality_EU_train[is.na(centrality_EU_train)] <- 0

centrality_Eu_test <- merge(eu_user_test,centrality, by="user_id", all.x = TRUE)
centrality_Eu_test <- centrality_Eu_test[,c(1,8:11)]
centrality_Eu_test[is.na(centrality_Eu_test)] <- 0



# Spanish

edeges_es <- read_csv("edeges_es.csv", col_types = cols(source = col_character(), 
                                                        target = col_character()))

load("D:/andre/Documenti/vaxxstance/rifatti/ES_user_test.Rdata")
load("D:/andre/Documenti/vaxxstance/rifatti/ES_label.Rdata")
load("D:/andre/Documenti/vaxxstance/rifatti/ES_label_train.Rdata")

es_label <- dati

es_label_train <- dato

names(edeges_es)[1]="Source"
names(edeges_es)[2]="Target"

esrt <- as.matrix(edeges_es)
g <- graph_from_edgelist(esrt[,1:2], directed=TRUE)
V(g)$label <- V(g)$name

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


centrality_ES_train <- merge(es_label_train,centrality, by="user_id", all.x = TRUE)
centrality_ES_train <- centrality_ES_train[,c(1,4:7)]
centrality_ES_train[is.na(centrality_ES_train)] <- 0

centrality_ES_test <- merge(es_user_test,centrality, by="user_id", all.x = TRUE)
centrality_ES_test <- centrality_ES_test[,c(1,8:11)]
centrality_ES_test[is.na(centrality_ES_test)] <- 0
