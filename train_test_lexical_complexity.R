# lexical complexity indices

rm(list=ls())

library(readr)
library(quanteda)

wdir <- "D:/andre/Documenti/vaxxstance/"
setwd( wdir )



# Spanish

# train

es_train <- read_csv("csv/es_train/es_train.csv", 
                     col_types = cols(tweet_id = col_character(), 
                                      user_id = col_character()))


es_corp <- quanteda::corpus(es_train$text)


# quanteda dtm, no stemming, no stopwords
es_dtm <- quanteda::dfm(es_corp,
                        remove_punct=TRUE,
                        remove_numbers=TRUE,
                        remove_symbols=TRUE,
                        remove_separators=TRUE,
                        remove_url=TRUE)


# lexical diversity and readability can act as proxies for textual complexity
########################## calculating readability
# Note that there are dozens of different readability measures
readab <- textstat_readability(es_corp,
                               measure=c("ARI","Bormuth.MC","Coleman","Coleman.C2","Dale.Chall", "Danielson.Bryan.2","FOG"))


########################## calculating diversity
# Can only be calculated for tokens or a sole_dtm
lexdiv <- textstat_lexdiv(es_dtm, 
                          measure=c("TTR","R","I"))


es_lex <- cbind(es_train[,c(1:2)],readab[,c(2:8)],lexdiv[,c(2:4)])




# test


es_test <- read_csv("es_test/es_test.csv", 
                    col_types = cols(tweet_id = col_character(), 
                                     user_id = col_character()))

es_corp <- quanteda::corpus(es_test$text)


# quanteda dtm, no stemming, no stopwords
es_dtm <- quanteda::dfm(es_corp,
                        remove_punct=TRUE,
                        remove_numbers=TRUE,
                        remove_symbols=TRUE,
                        remove_separators=TRUE,
                        remove_url=TRUE)


# lexical diversity and readability can act as proxies for textual complexity
########################## calculating readability
# Note that there are dozens of different readability measures
readab <- textstat_readability(es_corp,
                               measure=c("ARI","Bormuth.MC","Coleman","Coleman.C2","Dale.Chall", "Danielson.Bryan.2","FOG"))


########################## calculating diversity
# Can only be calculated for tokens or a sole_dtm
lexdiv <- textstat_lexdiv(es_dtm, 
                          measure=c("TTR","R","I"))


es_lex <- cbind(es_test[,c(1:2)],readab[,c(2:8)],lexdiv[,c(2:4)])



# lexical complexity

rm(list = ls())

# basque

# train

eu_train <- read_csv("csv/eu_train/eu_train.csv", 
                     col_types = cols(tweet_id = col_character(), 
                                      user_id = col_character()))


eu_corp <- quanteda::corpus(eu_train$text)


# quanteda dtm, no stemming, no stopwords
eu_dtm <- quanteda::dfm(eu_corp,
                        remove_punct=TRUE,
                        remove_numbers=TRUE,
                        remove_symbols=TRUE,
                        remove_separators=TRUE,
                        remove_url=TRUE)


# lexical diversity and readability can act as proxies for textual complexity
########################## calculating readability
# Note that there are dozens of different readability measures
readab <- textstat_readability(eu_corp,
                               measure=c("ARI","Bormuth.MC","Dale.Chall.old","Danielson.Bryan.2","Flesch", "FOG"))


########################## calculating diversity
# Can only be calculated for tokens or a sole_dtm
lexdiv <- textstat_lexdiv(eu_dtm, 
                          measure=c("R","I","D"))


eu_lex <- cbind(eu_train[,c(1:2)],readab[,c(2:7)],lexdiv[,c(2:4)])



# test
eu_test <- read_csv("eu_test/eu_test.csv", 
                    col_types = cols(tweet_id = col_character(), 
                                     user_id = col_character()))

eu_corp <- quanteda::corpus(eu_test$text)


# quanteda dtm, no stemming, no stopwords
eu_dtm <- quanteda::dfm(eu_corp,
                        remove_punct=TRUE,
                        remove_numbers=TRUE,
                        remove_symbols=TRUE,
                        remove_separators=TRUE,
                        remove_url=TRUE)


# lexical diversity and readability can act as proxies for textual complexity
########################## calculating readability
# Note that there are dozens of different readability measures
readab <- textstat_readability(eu_corp,
                               measure=c("ARI","Bormuth.MC","Dale.Chall.old","Danielson.Bryan.2","Flesch", "FOG"))


########################## calculating diversity
# Can only be calculated for tokens or a sole_dtm
lexdiv <- textstat_lexdiv(eu_dtm, 
                          measure=c("R","I","D"))


eu_lex <- cbind(eu_test[,c(1:2)],readab[,c(2:7)],lexdiv[,c(2:4)])

