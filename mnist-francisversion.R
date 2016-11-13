require(mxnet)
setwd("F:/Dropbox (aLV)/Francis/R-Test/R-Test")

#http://mxnet.io/get_started/index.html

train <- mx.io.MNISTIter(
  image       = "train-images-idx3-ubyte",
  label       = "train-labels-idx1-ubyte",
  input_shape = c(28, 28, 1),
  batch_size  = 100,
  shuffle     = TRUE,
  flat        = TRUE
)

val <- mx.io.MNISTIter(
  image       = "t10k-images-idx3-ubyte",
  label       = "t10k-labels-idx1-ubyte",
  input_shape = c(28, 28, 1),
  batch_size  = 100,
  flat        = TRUE)

data <- mx.symbol.Variable('data')
fc1  <- mx.symbol.FullyConnected(data = data, name = 'fc1', num_hidden = 256)
act1 <- mx.symbol.Activation(data = fc1, name = 'relu1', act_type = "relu")
fc2  <- mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
act2 <- mx.symbol.Activation(data = fc2, name = 'relu2', act_type = "relu")
fc3  <- mx.symbol.FullyConnected(data = act2, name = 'fc3', num_hidden = 10)
mlp  <- mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')

model <- mx.model.FeedForward.create(
  X                  = train,
  eval.data          = val,
  ctx                = mx.gpu(),
  symbol             = mlp,
  eval.metric        = mx.metric.accuracy,
  num.round          = 200,
  learning.rate      = 0.1,
  momentum           = 0.9,
  wd                 = 0.0001,
  array.batch.size   = 100,
  epoch.end.callback = mx.callback.save.checkpoint("mnist"),
  batch.end.callback = mx.callback.log.train.metric(50)
)

test <- mx.io.MNISTIter(
  image       = "t10k-images-idx3-ubyte",
  label       = "t10k-labels-idx1-ubyte",
  input_shape = c(28, 28, 1),
  batch_size  = 100,
  shuffle     = TRUE,
  flat        = TRUE
)

preds <- predict(model, test)

setwd("F:/Dropbox (aLV)/R/R Adventures/HandwritingRecognitionNNET/Data")
test <- read.csv('test.csv', header=TRUE)
test <- data.matrix(test)
test.array <- test
dim(test.array) <- c(28, 28, 1, ncol(test))


preds <- predict(model, test.array)
submission1101 <- data.frame(ImageId=1:ncol(test), Label=pred.label)
write.csv(submission1030, file='submission1101.csv', row.names=FALSE, quote=FALSE)


graph.viz(model$symbol$as.json())

#=====

test <- read.csv('test.csv', header=TRUE)


