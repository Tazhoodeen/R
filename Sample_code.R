####################################################################################
#    SCRIPT WHICH SOLVE POPULAR MACHINE LEARNING TASK - CLASSIFICATION PROBLEM     #
################################################ TAZHUDIN ALIEV ####################

#  install and  load libraries
install.packages("xgboost")
install.packages("caret")
library("xgboost")
library("caret")
# Set seed
set.seed(1)

# function to generate random proportions whose rowSums = 1 
props <- function(ncol, nrow, var.names=NULL){
  if (ncol < 2) stop("ncol must be greater than 1")
  p <- function(n){
    y <- 0
    z <- sapply(seq_len(n-1), function(i) {
      x <- sample(seq(0, 1-y, by=.01), 1)
      y <<- y + x
      return(x)
    }
    )
    w <- c(z , 1-sum(z))
    return(w)
  }
  DF <- data.frame(t(replicate(nrow, p(n=ncol))))
  if (!is.null(var.names)) colnames(DF) <- var.names
  return(DF)
}


# 1.Create dataframe
df <- props(ncol=20, nrow=2000)

# count number of rows
N <- nrow(df)

# 2. Create target variable
df$target <- 0

# random assign classes
df$target[sample(1:N,0.5*N)] <- 1

# 3. Train data

# Set parameters
ETA <- c(0.023)
DEPTH <- c(10)
SUBSAMPLE <- c(0.83)
COLSAMPLE <- c(0.83)
CHILDWEIGHT <- c(1)

# Data preparation and partition
h <- createDataPartition(df[,"target"],p=0.9)
h1 <- h[[1]]

# Plot histogram
hist(df[h1,"target"])

dval<-xgb.DMatrix(data=data.matrix(df[-h1,!(colnames(df) %in% c("target"))]),label=df$target[-h1])
dtrain<-xgb.DMatrix(data=data.matrix(df[h1,!(colnames(df) %in% c("target"))]),label=df$target[h1])
watchlist<-list(val=dval,train=dtrain)

# List of parameters
param <- list("objective" = "binary:logistic",
              "eval_metric" = "logloss",
              booster = "gbtree",
              eta                 = ETA, 
              max_depth           = DEPTH, 
              subsample           = SUBSAMPLE, 
              min_child_weight    = CHILDWEIGHT,
              colsample_bytree    = COLSAMPLE
              )

clf <- xgb.train(   params              = param,
                    data                = dtrain,
                    nrounds             = 300, 
                    verbose             = 0,
                    early.stop.round    = 50,
                    watchlist           = watchlist
                    
)



Best_iter <- clf[[3]]
Best_ind <- clf[[4]]
print(paste("Best iteration gives evaluation metric equals to",Best_iter))
