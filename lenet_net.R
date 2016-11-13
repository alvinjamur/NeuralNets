require(mxnet)

# laptop path
# setwd("~/Dropbox (aLV)/R/R Adventures/HandwritingRecognitionNNET/Data")

# macpro path
#setwd("/Volumes/Promise Pegasus/Dropbox (aLV)/R/R Adventures/HandwritingRecognitionNNET/Data")

#laplace path

setwd("F:/Dropbox (aLV)/R/R Adventures/HandwritingRecognitionNNET/Data")

train <- read.csv('train.csv', header=TRUE)
test <- read.csv('test.csv', header=TRUE)
train <- data.matrix(train)
test <- data.matrix(test)

train.x <- train[,-1]
train.y <- train[,1]

train.x <- t(train.x/255)
test <- t(test/255)

# NNET ARCHITECTURE HERE

# input
data <- mx.symbol.Variable('data')

# first conv
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# second conv
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# first fullc
flatten <- mx.symbol.Flatten(data=pool2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")

# second fullc
fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
# loss
lenet <- mx.symbol.SoftmaxOutput(data=fc2)

# Then let us reshape the matrices into arrays:
  
train.array <- train.x
dim(train.array) <- c(28, 28, 1, ncol(train.x))
test.array <- test
dim(test.array) <- c(28, 28, 1, ncol(test))

devices <- mx.cpu()

mx.set.seed(10536)

starttime <- proc.time()

model <- mx.model.FeedForward.create(lenet, X=train.array, y=train.y,
                                     ctx=devices, num.round=75, array.batch.size=100,
                                     learning.rate=0.05, momentum=0.9, wd=0.00001,
                                     eval.metric=mx.metric.accuracy,
                                     verbose=TRUE,
                                     epoch.end.callback=mx.callback.log.train.metric(100))

print(proc.time() - starttime)

preds <- predict(model, test.array)
pred.label <- max.col(t(preds)) - 1
submitlenet <- data.frame(ImageId=1:ncol(test), Label=pred.label)
write.csv(submitlenet, file='submitlenet.csv', row.names=FALSE, quote=FALSE)

to_graphviz(model)
