library(terra)

load('amt_fisher_covar.rda')
load('amt_fisher.rda')

landuse<-unwrap(amt_fisher_covar$landuse)
elevation<-unwrap(amt_fisher_covar$elevation)
popden<-unwrap(amt_fisher_covar$popden)

writeRaster(landuse,"landuse.tif",overwrite=TRUE)
writeRaster(elevation,"elevation.tif",overwrite=TRUE)
writeRaster(popden,"population.tif",overwrite=TRUE)

colnames(amt_fisher)<-c("X","Y","Date","sex","id","name")

amt_fisher$Date<-as.POSIXct(amt_fisher$Date)
amt_fisher$Date<-trunc(amt_fisher$Date,"secs")
write.csv(amt_fisher,"amt_fisher.csv",row.names=FALSE)
