# In-out label count - original network - friends

rm(list=ls())


library(readr)
library(reshape2)


# Basque

# load labelled data
load("D:/andre/Documenti/vaxxstance/rifatti/EU_label.Rdata")
load("D:/andre/Documenti/vaxxstance/rifatti/EU_label_train.Rdata")
load("D:/andre/Documenti/vaxxstance/rifatti/EU_user_test.Rdata")

# import trai-test sets
eu_friend_train <- read_csv("D:/andre/Documenti/vaxxstance/csv/eu_train/eu_friend_train.csv", 
                            col_types = cols(Source = col_character(), 
                                             Target = col_character()))
eu_friend_test <- read_csv("D:/andre/Documenti/vaxxstance/eu_test/eu_friend_test.csv", 
                           col_types = cols(Source = col_character(), 
                                            Target = col_character()))

eu_friend <- rbind(eu_friend_train,eu_friend_test)

eu_label <- dati[,1:2]


# retweet done
names(eu_friend)[2]="user_id"

eu_edges_lab <- merge(eu_friend,eu_label, by="user_id", all.x = TRUE)
names(eu_edges_lab)[3]="target_label"


eu_edges_lab <- eu_edges_lab[,c(2,3)]


done <- dcast(eu_edges_lab, Source~target_label, length)
names(done)[1]="user_id"
names(done)[2]="against_out"
names(done)[3]="favor_out"
names(done)[4]="none_out"

done_user <- merge(dato,done, by="user_id", all.x = TRUE)
done_user <- done_user[,c(1,4:6)]
done_user[is.na(done_user)] <- 0



# received
names(eu_label)[1]="Source"

eu_edges_lab_rec <- merge(eu_friend,eu_label, by="Source", all.x = TRUE)
names(eu_edges_lab_rec)[2]="target"
names(eu_edges_lab_rec)[3]="source_label"


eu_edges_lab_rec <- eu_edges_lab_rec[,c(2,3)]


receiv <- dcast(eu_edges_lab_rec, target~source_label, length)
names(receiv)[1]="user_id"
names(receiv)[2]="against_in"
names(receiv)[3]="favor_in"
names(receiv)[4]="none_in"

receiv_user <- merge(dato,receiv, by="user_id", all.x = TRUE)
receiv_user <- receiv_user[,c(1,4:6)]
receiv_user[is.na(receiv_user)] <- 0

train_EU_friends_orig_label_count <- merge(done_user,receiv_user, by="user_id")


done_user_test <- merge(eu_user_test,done, by="user_id", all.x = TRUE)
done_user_test <- done_user_test[,c(1,8:10)]
done_user_test[is.na(done_user_test)] <- 0

receiv_user_test <- merge(eu_user_test,receiv, by="user_id", all.x = TRUE)
receiv_user_test <- receiv_user_test[,c(1,8:10)]
receiv_user_test[is.na(receiv_user_test)] <- 0

test_EU_friends_orig_label_count <- merge(done_user_test,receiv_user_test, by="user_id")




# Spanish

load("D:/andre/Documenti/vaxxstance/rifatti/ES_label.Rdata")
load("D:/andre/Documenti/vaxxstance/rifatti/ES_label_train.Rdata")
load("D:/andre/Documenti/vaxxstance/rifatti/ES_user_test.Rdata")

ES_friend_train <- read_csv("D:/andre/Documenti/vaxxstance/csv/ES_train/ES_friend_train.csv", 
                            col_types = cols(Source = col_character(), 
                                             Target = col_character()))
ES_friend_test <- read_csv("D:/andre/Documenti/vaxxstance/ES_test/ES_friend_test.csv", 
                           col_types = cols(source = col_character(), 
                                            target = col_character()))

names(ES_friend_test)[1]="Source"
names(ES_friend_test)[2]="Target"
ES_friend <- rbind(ES_friend_train,ES_friend_test)

ES_label <- dati[,1:2]


# rt done
names(ES_friend)[2]="user_id"

ES_edges_lab <- merge(ES_friend,ES_label, by="user_id", all.x = TRUE)
names(ES_edges_lab)[3]="target_label"


ES_edges_lab <- ES_edges_lab[,c(2,3)]


done <- dcast(ES_edges_lab, Source~target_label, length)
names(done)[1]="user_id"
names(done)[2]="against_out"
names(done)[3]="favor_out"
names(done)[4]="none_out"

done_user <- merge(dato,done, by="user_id", all.x = TRUE)
done_user <- done_user[,c(1,4:6)]
done_user[is.na(done_user)] <- 0



# received
names(ES_label)[1]="Source"

ES_edges_lab_rec <- merge(ES_friend,ES_label, by="Source", all.x = TRUE)
names(ES_edges_lab_rec)[2]="target"
names(ES_edges_lab_rec)[3]="source_label"


ES_edges_lab_rec <- ES_edges_lab_rec[,c(2,3)]



receiv <- dcast(ES_edges_lab_rec, target~source_label, length)
names(receiv)[1]="user_id"
names(receiv)[2]="against_in"
names(receiv)[3]="favor_in"
names(receiv)[4]="none_in"

receiv_user <- merge(dato,receiv, by="user_id", all.x = TRUE)
receiv_user <- receiv_user[,c(1,4:6)]
receiv_user[is.na(receiv_user)] <- 0

train_ES_friends_orig_label_count <- merge(done_user,receiv_user, by="user_id")


done_user_test <- merge(es_user_test,done, by="user_id", all.x = TRUE)
done_user_test <- done_user_test[,c(1,8:10)]
done_user_test[is.na(done_user_test)] <- 0

receiv_user_test <- merge(es_user_test,receiv, by="user_id", all.x = TRUE)
receiv_user_test <- receiv_user_test[,c(1,8:10)]
receiv_user_test[is.na(receiv_user_test)] <- 0

test_ES_friends_orig_label_count <- merge(done_user_test,receiv_user_test, by="user_id")



