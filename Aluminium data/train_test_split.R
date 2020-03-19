train_test_split <- function(df){
  smp_size <- floor(0.75* nrow(df))
  
  set.seed(123)
  train_ind <- sample(seq_len(nrow(df)), size = smp_size)
  
  train <- df[train_ind, ]
  test <- df[-train_ind, ]
  
  return (list(train,test))
}