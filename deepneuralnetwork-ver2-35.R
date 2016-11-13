library(readr)

# point to where the kaggle data sets are, then read 'em in

setwd("~/Dropbox (aLV)/R/R Adventures/HandwritingRecognitionNNET/Data")
train <- read_csv("train.csv")
test <- read_csv("test.csv")

# look at data

head(train[1:10])

# The first column is the actual digit which we'll convert to a factor a 'lil later 
# Create a 28*28 matrix with pixel color values and plot 'em

m = matrix(unlist(train[10,-1]),nrow = 28,byrow = T)
image(m,col=grey.colors(255))

# Lets rotate the matrix to see the image properly

rotate <- function(x) t(apply(x, 2, rev))

# Now plot a bunch of images

par(mfrow=c(2,3))
lapply(1:6, 
       function(x) image(
         rotate(matrix(unlist(train[x,-1]),nrow = 28,byrow = T)),
         col=grey.colors(255),
         xlab=train[x,1]
       )
)

par(mfrow=c(1,1)) 

# Now, start h2o locally, using 12GB of ram and all processors

library(h2o)
localH2O = h2o.init(max_mem_size = '12g', 
                    nthreads = -1)

# convert those digit labels to factor for classification and
# then save into h2o framework

train[,1] = as.factor(train$label) 
train_h2o = as.h2o(train)
test_h2o = as.h2o(test)

# set timer

s <- proc.time()

# here we train the neural network 
model =
  h2o.deeplearning(x = 2:785,                                      # predictor columns
                   y = 1,                                          # column number for the label
                   training_frame = train_h2o,                     # data in H2O format
                   activation = "RectifierWithDropout",            # choose transfer function here
                   input_dropout_ratio = 0.2,                      # % of inputs dropout
                   hidden_dropout_ratios = c(0.01,0.01,0.01,0.01), # % for nodes dropout
                   balance_classes = TRUE,                         # gotta balance those classes
                   hidden = c(100,70,10,30),                       # layers with neurons
                   momentum_stable = 0.99,
                   nesterov_accelerated_gradient = T,              # use this for speed
                   epochs = 20)                                    # no. of epochs

# print confusion matrix

h2o.confusionMatrix(model)

# print time elapsed

print("Time elapsed to train network and print matrix....")
s - proc.time()

# Now onto classifying test set

h2o_y_test <- h2o.predict(model, test_h2o)

# convert H2O format into data frame and  save as csv

df_y_test = as.data.frame(h2o_y_test)
df_y_test = data.frame(ImageId = seq(1,length(df_y_test$predict)), Label = df_y_test$predict)
write.csv(df_y_test, file = "submission-r-h2o.csv", row.names=F)

# shut down virutal H2O cluster

h2o.shutdown(prompt = F)



