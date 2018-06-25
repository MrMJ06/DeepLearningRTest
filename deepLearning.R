anomaly_model <- h2o.deeplearning(x = names(train_ecg),
                                  training_frame = train_ecg,
                                  activation = "Tanh",
                                  autoencoder = TRUE,
                                  hidden = c(50,20,50),
                                  l1 = 1e-4,
                                  epochs = 100)

recon_error <- h2o.anomaly(anomaly_model, test_ecg)

recon_error <- as.data.frame(recon_error)
recon_error
plot.ts(recon_error)

MNIST_DIGITStrain = read.csv('train.csv')

#Inspeccionamos el data set
dim(MNIST_DIGITStrain)
head(MNIST_DIGITStrain)
MNIST_DIGITStrain[1:5, 1:7]

par( mfrow = c(10,10), mai = c(0,0,0,0)) # Indica que el plot es de 10 x 10 imagenes con la función c
for(i in 1:100){ #Bucle para recorrer las 100 imágenes
  y = as.matrix(MNIST_DIGITStrain[i,2:785]) # Accede a la fila i y las columnas de la 2 a última(quita label)
  dim(y) = c(28, 28) #
  image( y[,nrow(y):1], axes = FALSE, col = gray(255:0 / 255))
  text( 0.2, 0, MNIST_DIGITStrain[i,1], cex = 3, col = 2, pos = c(3,4))
}

mfile = 'train.csv'
MDIG = h2o.importFile(path=mfile,sep=',')

# Show the data objects on the H2O platform
# h2o.ls()


NN_model = h2o.deeplearning(
  x = 2:785,
  training_frame = MDIG,
  hidden = c(400,200,2,200,400),
  epochs = 100,
  activation = 'Tanh',
  autoencoder = TRUE
)

train_supervised_features2 = h2o.deepfeatures(NN_model,MDIG, layer=3)

plotdata2 = as.data.frame(train_supervised_features2)
plotdata2$label = as.character(as.vector(MDIG[,1]))

qplot(DF.L3.C1, DF.L3.C2, data = plotdata2, color = label, main = 'Neural network: 400 - 200 - 2 - 200 - 400')
