# In-out label count - augmented network - retweets

rm(list=ls())


library(readr)
library(reshape2)

# Basque

# load labelled data
load("D:/andre/Documenti/vaxxstance/rifatti/EU_label.Rdata")
load("D:/andre/Documenti/vaxxstance/rifatti/EU_label_train.Rdata")
load("D:/andre/Documenti/vaxxstance/rifatti/EU_user_test.Rdata")

# import augmented nework
edeges_eu <- read_csv("edeges_eu.csv", col_types = cols(source = col_character(), 
                                                        target = col_character()))


eu_label <- dati[,1:2]


# retweet done
names(edeges_eu)[2]="user_id"

eu_edges_lab <- merge(edeges_eu,eu_label, by="user_id", all.x = TRUE)
names(eu_edges_lab)[3]="target_label"


eu_edges_lab <- eu_edges_lab[,c(2,3)]


done <- dcast(eu_edges_lab, source~target_label, length)
names(done)[1]="user_id"
names(done)[2]="against_done"
names(done)[3]="favor_done"
names(done)[4]="none_done"

done_user <- merge(dato,done, by="user_id", all.x = TRUE)
done_user <- done_user[,c(1,4:6)]
done_user[is.na(done_user)] <- 0



# received
names(eu_label)[1]="source"

eu_edges_lab_rec <- merge(edeges_eu,eu_label, by="source", all.x = TRUE)
names(eu_edges_lab_rec)[2]="target"
names(eu_edges_lab_rec)[3]="source_label"


eu_edges_lab_rec <- eu_edges_lab_rec[,c(2,3)]


receiv <- dcast(eu_edges_lab_rec, target~source_label, length)
names(receiv)[1]="user_id"
names(receiv)[2]="against_receiv"
names(receiv)[3]="favor_receiv"
names(receiv)[4]="none_receiv"

receiv_user <- merge(dato,receiv, by="user_id", all.x = TRUE)
receiv_user <- receiv_user[,c(1,4:6)]
receiv_user[is.na(receiv_user)] <- 0

train_EU_rt_augm_label_count <- merge(done_user,receiv_user, by="user_id")


done_user_test <- merge(eu_user_test,done, by="user_id", all.x = TRUE)
done_user_test <- done_user_test[,c(1,8:10)]
done_user_test[is.na(done_user_test)] <- 0

receiv_user_test <- merge(eu_user_test,receiv, by="user_id", all.x = TRUE)
receiv_user_test <- receiv_user_test[,c(1,8:10)]
receiv_user_test[is.na(receiv_user_test)] <- 0

test_EU_rt_augm_label_count <- merge(done_user_test,receiv_user_test, by="user_id")





# Spanish
# see basque for details

load("D:/andre/Documenti/vaxxstance/rifatti/ES_label.Rdata")
load("D:/andre/Documenti/vaxxstance/rifatti/ES_label_train.Rdata")
load("D:/andre/Documenti/vaxxstance/rifatti/ES_user_test.Rdata")


edeges_es <- read_csv("D:/andre/Documenti/vaxxstance/edeges_es.csv", col_types = cols(source = col_character(), 
                                                                                      target = col_character()))

# rt done
names(edeges_es)[2]="user_id"

es_label <- dati[,1:2]

es_edges_lab <- merge(edeges_es,es_label, by="user_id", all.x = TRUE)
names(es_edges_lab)[3]="target_label"


es_edges_lab <- es_edges_lab[,c(2,3)]


done <- dcast(es_edges_lab, source~target_label, length)
names(done)[1]="user_id"
names(done)[2]="against_done"
names(done)[3]="favor_done"
names(done)[4]="none_done"

done_user <- merge(dato,done, by="user_id", all.x = TRUE)
done_user <- done_user[,c(1,4:6)]
done_user[is.na(done_user)] <- 0


# received
names(es_label)[1]="source"

es_edges_lab_rec <- merge(edeges_es,es_label, by="source", all.x = TRUE)
names(es_edges_lab_rec)[2]="target"
names(es_edges_lab_rec)[3]="source_label"


es_edges_lab_rec <- es_edges_lab_rec[,c(2,3)]


receiv <- dcast(es_edges_lab_rec, target~source_label, length)
names(receiv)[1]="user_id"
names(receiv)[2]="against_receiv"
names(receiv)[3]="favor_receiv"
names(receiv)[4]="none_receiv"


receiv_user <- merge(dato,receiv, by="user_id", all.x = TRUE)
receiv_user <- receiv_user[,c(1,4:6)]
receiv_user[is.na(receiv_user)] <- 0

train_ES_rt_augm_label_count <- merge(done_user,receiv_user, by="user_id")


done_user_test <- merge(es_user_test,done, by="user_id", all.x = TRUE)
done_user_test <- done_user_test[,c(1,8:10)]
done_user_test[is.na(done_user_test)] <- 0

receiv_user_test <- merge(es_user_test,receiv, by="user_id", all.x = TRUE)
receiv_user_test <- receiv_user_test[,c(1,8:10)]
receiv_user_test[is.na(receiv_user_test)] <- 0

test_ES_rt_augm_label_count <- merge(done_user_test,receiv_user_test, by="user_id")


