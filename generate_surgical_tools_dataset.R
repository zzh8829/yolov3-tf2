library(tidyverse)

data_dir <- '~/Documents/USC/USC_docs/ml/clean-surgical-ds/'
surgical_images_all <- list.files(file.path(data_dir, 'JPEGImages')) %>%
  gsub('\\.jpe?g$', '', .)

# Make training
data_set_surgical <- data.frame(i=surgical_images_all, stringsAsFactors = F) %>%
  mutate(X1=ifelse(grepl('^S611T1', i), 'validation', 'train'))

data_set_surgical %>%
  filter(X1 == 'train') %>%
  dplyr::select(i) %>%
  # head(100) %>%
  mutate(v=1) %>%
  write.table(., file = file.path(data_dir, 'ImageSetS/Main/surgical_train.txt'), sep='\t', quote=F, row.names=F, col.names = F)

data_set_surgical %>%
  filter(X1 == 'validation') %>%
  dplyr::select(i) %>%
  # head(100) %>%
  mutate(v=1) %>%
  write.table(., file = file.path(data_dir, 'ImageSetS/Main/surgical_val.txt'), sep='\t', quote=F, row.names=F, col.names = F)

table(data_set_surgical$X1)
