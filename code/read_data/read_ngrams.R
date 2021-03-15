library(ngramr)
library(dplyr)

words <- read.csv("../../data/processed/allwords.csv", encoding="UTF-8")

words <- pull(words, "word")

words <- as.character(words)

df_all <- data.frame()

for (i in seq(1, 5000, 12)) {
  subset <- words[i:min(5000, i+12)]
  df_subset <- ngram(subset)
  df_all <- rbind(df_all, df_subset)
}

write.csv(df_all, "")