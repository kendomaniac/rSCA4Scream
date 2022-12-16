docker build --platform linux/amd64 . -t  repbioinfo/train_autoencoder_cpu
@Set "Build=%CD%"
@Echo(%Build%
@If Not Exist "configurationFile.txt" Set /P "=%Build%" 0<NUL 1>"configurationFile.txt"
mkdir %Build%
copy configurationFile.txt %Build%
del %Build%\id.txt
docker run --platform linux/amd64 -itv %Build%:/scratch -v /var/run/docker.sock:/var/run/docker.sock --cidfile  %Build%\id.txt repbioinfo/train_autoencoder_cpu
