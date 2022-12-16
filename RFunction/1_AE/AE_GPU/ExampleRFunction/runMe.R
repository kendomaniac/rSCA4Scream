source("runDocker.R")
source("autoencoderGPU.R")
scratch.folder=getwd()
file=paste(scratch.folder,"setA.csv",sep="/")
separator=","
bias="kegg"
permutation=2
nEpochs=2000
projectName="Test_GPU"
autoencoder4clusteringGPU(group=c("docker"), scratch.folder=scratch.folder, file=file,separator=separator, bias=bias, permutation=permutation, nEpochs=nEpochs,projectName=projectName)
