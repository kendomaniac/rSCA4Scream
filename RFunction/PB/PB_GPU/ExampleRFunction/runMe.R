source("runDocker.R")
source("pseudoBulkGPU.R")
scratch.folder=getwd()
file=paste(scratch.folder,"latentSpace.csv",sep="/")
bn=paste(scratch.folder,"clustering.output.csv",sep="/")

separator=","
permutation=40
nEpochs=1000
projectName="Test_CPU"
nCluster=3
autoencoder4pseudoBulkGPU(group=c("docker"), scratch.folder=scratch.folder, file=file,separator=",", permutation=5, nEpochs=1000,projectName=projectName,bN=bn)
