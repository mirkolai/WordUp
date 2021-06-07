# In-out label count - original network - retweets

rm(list=ls())

# Basque

library(readr)
library(tidyverse)
library(tidyr)


# load labelled data
load("D:/andre/Documenti/vaxxstance/rifatti/EU_label.Rdata")
load("D:/andre/Documenti/vaxxstance/rifatti/EU_label_train.Rdata")
load("D:/andre/Documenti/vaxxstance/rifatti/EU_user_test.Rdata")

# import train-test set
eu_rt_train <- read_csv("D:/andre/Documenti/vaxxstance/csv/eu_train/eu_retweet_train.csv",
                        col_types = cols(Source = col_character(),
                                         Target = col_character()))
eu_rt_timel_test <- read_csv("D:/andre/Documenti/vaxxstance/eu_test/eu_retweet_timeline_test.csv",
                             col_types = cols(Source = col_character(),
                                              Target = col_character()))

eu_rt_test <- eu_rt_timel_test %>% filter(
  Target %in% eu_user_test$user_id
)


eu_rt_train$set <- "train"
eu_rt_test$set <- "test"

eu_rt <- rbind(eu_rt_train[,c(1:4)],eu_rt_test[,c(1:4)])

eu_label <- dati[,1:2]


# retweet done
names(eu_rt)[2]="user_id"

eu_edges_lab <- merge(eu_rt,eu_label, by="user_id", all.x = TRUE)
names(eu_edges_lab)[5]="target_label"


count_lab_eu <- eu_edges_lab %>%
  count(eu_edges_lab$Source, target_label, wt = Weight)
names(count_lab_eu)[1]="Source"


count_eu_fav <- count_lab_eu %>%
  group_by(Source) %>% 
  filter(target_label=="FAVOR") %>%
  summarize(favor_done = sum(n))
  
  
count_eu_ag <- count_lab_eu %>%
  group_by(Source) %>% 
  filter(target_label=="AGAINST") %>%
  summarize(against_done = sum(n))

count_eu_no <- count_lab_eu %>%
  group_by(Source) %>% 
  filter(target_label=="NONE") %>%
  summarize(none_done = sum(n))
  

names(count_eu_fav)[1]="user_id"
names(count_eu_ag)[1]="user_id"
names(count_eu_no)[1]="user_id"



done_user <- merge(dato,count_eu_fav, by="user_id", all.x = TRUE)
done_user <- merge(done_user,count_eu_ag, by="user_id", all.x = TRUE)
done_user <- merge(done_user,count_eu_no, by="user_id", all.x = TRUE)

done_user <- done_user[,c(1,4:6)]
done_user[is.na(done_user)] <- 0



# received
names(eu_label)[1]="Source"

eu_edges_lab_rec <- merge(eu_rt,eu_label, by="Source", all.x = TRUE)
names(eu_edges_lab_rec)[2]="target"
names(eu_edges_lab_rec)[5]="source_label"


eu_edges_lab_rec <- eu_edges_lab_rec[,c(2,3,5)]


count_lab_eu_rec <- eu_edges_lab_rec %>%
  count(target, source_label, wt = Weight)


count_eu_fav_rec <- count_lab_eu_rec %>%
  group_by(target) %>% 
  filter(source_label=="FAVOR") %>%
  summarize(favor_receiv = sum(n))


count_eu_ag_rec <- count_lab_eu_rec %>%
  group_by(target) %>% 
  filter(source_label=="AGAINST") %>%
  summarize(against_receiv = sum(n))

count_eu_no_rec <- count_lab_eu_rec %>%
  group_by(target) %>% 
  filter(source_label=="NONE") %>%
  summarize(none_receiv = sum(n))


names(count_eu_fav_rec)[1]="user_id"
names(count_eu_ag_rec)[1]="user_id"
names(count_eu_no_rec)[1]="user_id"



receiv_user <- merge(dato,count_eu_fav_rec, by="user_id", all.x = TRUE)
receiv_user <- merge(receiv_user,count_eu_ag_rec, by="user_id", all.x = TRUE)
receiv_user <- merge(receiv_user,count_eu_no_rec, by="user_id", all.x = TRUE)

receiv_user <- receiv_user[,c(1,4:6)]
receiv_user[is.na(receiv_user)] <- 0


train_EU_rt_orig_label_count <- merge(done_user,receiv_user, by="user_id")


done_user_test <- merge(eu_user_test,count_eu_fav, by="user_id", all.x = TRUE)
done_user_test <- merge(done_user_test,count_eu_ag, by="user_id", all.x = TRUE)
done_user_test <- merge(done_user_test,count_eu_no, by="user_id", all.x = TRUE)
done_user_test <- done_user_test[,c(1,8:10)]
done_user_test[is.na(done_user_test)] <- 0

receiv_user_test <- merge(eu_user_test,count_eu_fav_rec, by="user_id", all.x = TRUE)
receiv_user_test <- merge(receiv_user_test,count_eu_ag_rec, by="user_id", all.x = TRUE)
receiv_user_test <- merge(receiv_user_test,count_eu_no_rec, by="user_id", all.x = TRUE)
receiv_user_test <- receiv_user_test[,c(1,8:10)]
receiv_user_test[is.na(receiv_user_test)] <- 0

test_EU_rt_orig_label_count <- merge(done_user_test,receiv_user_test, by="user_id")




# Spanish
# see basque for more details

load("D:/andre/Documenti/vaxxstance/rifatti/ES_label.Rdata")
load("D:/andre/Documenti/vaxxstance/rifatti/ES_label_train.Rdata")
load("D:/andre/Documenti/vaxxstance/rifatti/ES_user_test.Rdata")

ES_rt_train <- read_csv("D:/andre/Documenti/vaxxstance/csv/ES_train/ES_retweet_train.csv",
                        col_types = cols(Source = col_character(),
                                         Target = col_character()))
ES_rt_test <- read_csv("D:/andre/Documenti/vaxxstance/ES_test/ES_retweet_test.csv",
                             col_types = cols(Source = col_character(),
                                              Target = col_character()))


names(ES_rt_test)[1]="Source"
names(ES_rt_test)[2]="Target"
names(ES_rt_test)[3]="Weight"


ES_rt_train$set <- "train"
ES_rt_test$set <- "test"

ES_rt <- rbind(ES_rt_train[,c(1:4)],ES_rt_test[,c(1:4)])

ES_label <- dati[,1:2]


# retweet done
names(ES_rt)[2]="user_id"

ES_edges_lab <- merge(ES_rt,ES_label, by="user_id", all.x = TRUE)
names(ES_edges_lab)[5]="target_label"


count_lab_ES <- ES_edges_lab %>%
  count(ES_edges_lab$Source, target_label, wt = Weight)
names(count_lab_ES)[1]="Source"


count_ES_fav <- count_lab_ES %>%
  group_by(Source) %>% 
  filter(target_label=="FAVOR") %>%
  summarize(favor_done = sum(n))


count_ES_ag <- count_lab_ES %>%
  group_by(Source) %>% 
  filter(target_label=="AGAINST") %>%
  summarize(against_done = sum(n))

count_ES_no <- count_lab_ES %>%
  group_by(Source) %>% 
  filter(target_label=="NONE") %>%
  summarize(none_done = sum(n))


names(count_ES_fav)[1]="user_id"
names(count_ES_ag)[1]="user_id"
names(count_ES_no)[1]="user_id"



done_user <- merge(dato,count_ES_fav, by="user_id", all.x = TRUE)
done_user <- merge(done_user,count_ES_ag, by="user_id", all.x = TRUE)
done_user <- merge(done_user,count_ES_no, by="user_id", all.x = TRUE)

done_user <- done_user[,c(1,4:6)]
done_user[is.na(done_user)] <- 0



# received
names(ES_label)[1]="Source"

ES_edges_lab_rec <- merge(ES_rt,ES_label, by="Source", all.x = TRUE)
names(ES_edges_lab_rec)[2]="target"
names(ES_edges_lab_rec)[5]="source_label"


ES_edges_lab_rec <- ES_edges_lab_rec[,c(2,3,5)]



count_lab_ES_rec <- ES_edges_lab_rec %>%
  count(target, source_label, wt = Weight)


count_ES_fav_rec <- count_lab_ES_rec %>%
  group_by(target) %>% 
  filter(source_label=="FAVOR") %>%
  summarize(favor_receiv = sum(n))


count_ES_ag_rec <- count_lab_ES_rec %>%
  group_by(target) %>% 
  filter(source_label=="AGAINST") %>%
  summarize(against_receiv = sum(n))

count_ES_no_rec <- count_lab_ES_rec %>%
  group_by(target) %>% 
  filter(source_label=="NONE") %>%
  summarize(none_receiv = sum(n))


names(count_ES_fav_rec)[1]="user_id"
names(count_ES_ag_rec)[1]="user_id"
names(count_ES_no_rec)[1]="user_id"



receiv_user <- merge(dato,count_ES_fav_rec, by="user_id", all.x = TRUE)
receiv_user <- merge(receiv_user,count_ES_ag_rec, by="user_id", all.x = TRUE)
receiv_user <- merge(receiv_user,count_ES_no_rec, by="user_id", all.x = TRUE)

receiv_user <- receiv_user[,c(1,4:6)]
receiv_user[is.na(receiv_user)] <- 0


train_ES_rt_orig_label_count <- merge(done_user,receiv_user, by="user_id")


done_user_test <- merge(es_user_test,count_ES_fav, by="user_id", all.x = TRUE)
done_user_test <- merge(done_user_test,count_ES_ag, by="user_id", all.x = TRUE)
done_user_test <- merge(done_user_test,count_ES_no, by="user_id", all.x = TRUE)
done_user_test <- done_user_test[,c(1,8:10)]
done_user_test[is.na(done_user_test)] <- 0

receiv_user_test <- merge(es_user_test,count_ES_fav_rec, by="user_id", all.x = TRUE)
receiv_user_test <- merge(receiv_user_test,count_ES_ag_rec, by="user_id", all.x = TRUE)
receiv_user_test <- merge(receiv_user_test,count_ES_no_rec, by="user_id", all.x = TRUE)
receiv_user_test <- receiv_user_test[,c(1,8:10)]
receiv_user_test[is.na(receiv_user_test)] <- 0

test_ES_rt_orig_label_count <- merge(done_user_test,receiv_user_test, by="user_id")

