#B565 Final Project by Ao Li
library(caret);library(randomForest);library(e1071);library(kernlab);library(doMC);library(foreach);library(RColorBrewer)
require(car);require(Hmisc);require(corrplot);require(neuralnet);require(plyr);require(flux)
setwd("~/Downloads/AoLi_FinalB565/data")

trFC <- read.csv('train_FNC.csv')
trSBM <- read.csv('train_SBM.csv')
dim(trFC)
dim(trSBM)

#functional

summary(trFC[,c(2:16)])
png("FNC.png", width=9, height=7, units="in", res=300)
par(mfrow=c(1,2))
boxplot(trFC[,c(2:16)],notch=T,col=c(2:16),main='Boxplots of FNC(1-15)')
grid(10, 10, lwd = 2)
hist(trFC[,3],col=rgb(1,0,0,0.5),breaks=20,prob=T,xlab='',ylab='',main='')
lines(density(trFC[,3]),lwd=3,col=rgb(1,0,0,0.7),lty=5)
hist(trFC[,4],col=rgb(0,0.5,0,0.5),add=T,breaks=20,prob=T)
lines(density(trFC[,4]),lwd=3,col=rgb(0,0.5,0,0.7),lty=5)
hist(trFC[,5],col=rgb(0,0,1,0.5),add=T,breaks=20,prob=T)
lines(density(trFC[,5]),lwd=3,col=rgb(0,0,0.7,0.5),lty=5)
title('Histograms for FNC(1-3)')
dev.off()


png("FNC_scatter.png", width=9, height=7, units="in", res=300)
scatterplotMatrix(trFC[,2:4])
dev.off()
#pearson's correlation
png("FNC_heat.png", width=9, height=7, units="in", res=300)
M<-rcorr(as.matrix(trFC[,2:16]), type="pearson")$r
corrplot(M, method = "circle")
dev.off()

#discuss about h-clustering
palette <- colorRampPalette(c('#f0f3ff','#0033BB'))(256)
heatmap(M,col=palette,symm=T,scale="column",Colv=NA)

#structural
summary(trSBM[,c(2:16)])
png("SBM.png", width=9, height=7, units="in", res=300)
par(mfrow=c(1,2))
boxplot(trSBM[,c(2:16)],notch=T,col=c(1:15),main='Boxplot for SBM(1-15)')
grid(10, 10, lwd = 2)

hist(trSBM[,3],col=rgb(1,0,0,0.5),breaks=20,prob=T,xlab='',ylab='',main='')
lines(density(trSBM[,3]),lwd=3,col=rgb(1,0,0,0.7),lty=5)
hist(trSBM[,4],col=rgb(0,0.5,0,0.5),add=T,breaks=20,prob=T)
lines(density(trSBM[,4]),lwd=3,col=rgb(0,0.5,0,0.7),lty=5)
hist(trSBM[,5],col=rgb(0,0,1,0.5),add=T,breaks=20,prob=T)
lines(density(trSBM[,5]),lwd=3,col=rgb(0,0,0.7,0.5),lty=5)
title('Histograms for SBM(1-3)')
dev.off()


png("SBM_scatter.png", width=9, height=7, units="in", res=300)
scatterplotMatrix(trSBM[,2:4])
dev.off()
#pearson's correlation
png("SBM_heat.png", width=9, height=7, units="in", res=300)
M<-rcorr(as.matrix(trSBM[,2:16]), type="pearson")$r
corrplot(M, method = "circle")
dev.off()

scatterplotMatrix(trSBM[,2:16])
#pearson's correlation
M<-rcorr(as.matrix(trSBM[,2:16]), type="pearson")$r
corrplot(M, method = "circle")
#discuss about h-clustering
palette <- colorRampPalette(c('#f0f3ff','#0033BB'))(256)
heatmap(M,col=palette,symm=T,scale="column",Colv=NA)



tr <- merge(trFC, trSBM, by='Id')
dim(tr)


M=as.matrix(dist(tr[,-1]))
dim(M)


# Determine number of clusters
wss <- (nrow(tr[,-1])-1)*sum(apply(tr[,-1],2,var))
for (i in 2:20) {
  wss[i] <- sum(kmeans(tr[,-1],centers=i,nstart=25, iter.max=1000)$withinss)
}
png("kmeans1.png", width=9, height=7, units="in", res=300)
plot(1:20, wss, type="b",lwd=3, xlab="Number of Clusters",ylab="Within groups sum of squares")
grid(10, 10, lwd = 2)
title('k-means clustering for features')
dev.off()

centers=6
cl<-kmeans(M, centers, iter.max = 100, nstart = 5)

png("kmeans2.png", width=9, height=7, units="in", res=300)
plot(M[,c(2,5)], col = cl$cluster,cex=1.5,pch=cl$cluster)
title('Clustering analysis')
text(M[,c(2,5)]+0.5, labels=colnames(M),col = cl$cluster)
grid(10, 10, lwd = 2)
dev.off()

# Cluster sizes
sort(table(cl$clust))
clust <- names(sort(table(cl$clust)))

y <- read.csv('train_labels.csv')
table(y[,2])

# First cluster
row.names(M[cl$clust==clust[1],])
row1<-row.names(M[cl$clust==clust[1],])
#User ID
y[row1,2]
# Second Cluster
row.names(M[cl$clust==clust[2],])
row2<-row.names(M[cl$clust==clust[2],])
#User ID
y[row2,2]
# Third Cluster
row.names(M[cl$clust==clust[3],])
row3<-row.names(M[cl$clust==clust[3],])
#User ID
y[row3,2]
# Fourth Cluster
row.names(M[cl$clust==clust[4],])
row4<-row.names(M[cl$clust==clust[4],])
#User ID
y[row4,2]
# Fifth Cluster
row.names(M[cl$clust==clust[5],])
row5<-row.names(M[cl$clust==clust[5],])
#User ID
y[row5,2]

row.names(M[cl$clust==clust[6],])




M=as.matrix(dist(t(tr[,-1])))
dim(M)

# Determine number of clusters
wss <- (nrow(tr[,-1])-1)*sum(apply(tr[,-1],2,var))
for (i in 2:20) {
  wss[i] <- sum(kmeans(tr[,-1],centers=i,nstart=25, iter.max=1000)$withinss)
}
png("kmeans1.png", width=9, height=7, units="in", res=300)
plot(1:20, wss, type="b",lwd=3, xlab="Number of Clusters",ylab="Within groups sum of squares")
grid(10, 10, lwd = 2)
title('k-means clustering for features')
dev.off()

centers=6
cl<-kmeans(M, centers, iter.max = 100, nstart = 5)

png("kmeans2.png", width=9, height=7, units="in", res=300)
plot(M[,c(2,5)], col = cl$cluster,cex=1.5,pch=cl$cluster)
title('Clustering analysis')
text(M[,c(2,5)]+0.5, labels=colnames(M),col = cl$cluster)
grid(10, 10, lwd = 2)
dev.off()

# Cluster sizes
sort(table(cl$clust))
clust <- names(sort(table(cl$clust)))

y <- read.csv('train_labels.csv')
table(y[,2])

# First cluster
row.names(M[cl$clust==clust[1],])

# Second Cluster
row.names(M[cl$clust==clust[2],])

# Third Cluster
row.names(M[cl$clust==clust[3],])

# Fourth Cluster
row.names(M[cl$clust==clust[4],])

# Fifth Cluster
row.names(M[cl$clust==clust[5],])


row.names(M[cl$clust==clust[6],])
#clustering does not make sense
#cluster of users, first two make sense
#cluster of regions, some sort of sense
================
#split the data into testing and training, 64 for training, 22 for testing
set.seed(3)
index <- sample(1:nrow(tr),round(0.75*nrow(tr)))
train <- tr[index,-1]
test <- tr[-index,-1]
y.train<-y[index,-1]
y.test<-y[-index,-1]

length(y.test)
length(y.train)
table(y.train)
table(y.test)

attach(y)


#logistic

datain.train<-cbind(train,y.train)
n <- names(datain.train)

f <- as.formula(paste("y.train ~", paste(n[!n %in% "y.train"], collapse = " + ")))

glm.fit=glm(f,data=datain.train,family=binomial)
summary(glm.fit)

glm.probs=predict(glm.fit,test,type="response")
glm.pred=rep(0,22)
glm.pred[glm.probs>.5]=1
table(glm.pred,y.test)
mean(glm.pred==y.test)

#roc curve

S=glm.probs;Y=y.test
roc.curve=function(s,print=FALSE){
  Ps=(S>s)*1
  FP=sum((Ps==1)*(Y==0))/sum(Y==0)
  TP=sum((Ps==1)*(Y==1))/sum(Y==1)
  if(print==TRUE){
    print(table(Observed=Y,Predicted=Ps))
  }
  vect=c(FP,TP)
  names(vect)=c("FPR","TPR")
  return(vect)
}
threshold = 0.5
roc.curve(threshold,print=TRUE)
mean(glm.pred==y.test)
ROC.curve=Vectorize(roc.curve)

M.ROC=ROC.curve(seq(0,1,by=.01))
png("ROCl.png", width=9, height=7, units="in", res=300)
plot(M.ROC[1,],M.ROC[2,],col="black",lwd=4,type="l",ylab='TPR',xlab='FPR',main='ROC')
abline(0,1,col='red',lwd=4)
grid(10, 10, lwd = 2)
dev.off()

auc(M.ROC[1,],M.ROC[2,])

#random forest
y <- read.csv('train_labels.csv')
trFC <- read.csv('train_FNC.csv')
trSBM <- read.csv('train_SBM.csv')
tr <- merge(trFC, trSBM, by='Id')

registerDoMC(cores=6)
y1 <- as.factor(paste('X.', y[,2], sep = ''))

rf.mod <- foreach(ntree=rep(2500, 6),.combine=combine,.multicombine=TRUE,
                  .packages='randomForest') %dopar% {
                    randomForest(tr[,-1], y1, ntree=ntree)
                  }

print(rf.mod)
png("RandomForest.png", width=9, height=7, units="in", res=300)
op = par(mfrow=c(1,2))
varImpPlot(rf.mod)
imp <- as.data.frame(rf.mod$importance[order(rf.mod$importance,decreasing=T),])
barplot(t(imp))
points(13,0.6, col=color[2], type='h', lwd=2)
title('Importance of features')
grid(10, 10, lwd = 2)
dev.off()

#neural network
rownames(imp)[c(1:13)]
ten<-rownames(imp)[c(1:13)]
dat.log <- tr[,ten]
colnames(dat.log)
set.seed(3)
index <- sample(1:nrow(tr),round(0.75*nrow(tr)))
attach(y)
datain<-dat.log
maxs <- apply(datain, 2, max)
mins <- apply(datain, 2, min)
scaled <- as.data.frame(scale(datain, center = mins, scale = maxs - mins))
train_ <- scaled[index,]
test_ <- scaled[-index,]
y.train<-y[index,2]
y.test<-y[-index,2]


maxs <- apply(datain, 2, max)
mins <- apply(datain, 2, min)
scaled <- as.data.frame(scale(datain, center = mins, scale = maxs - mins))
train_ <- scaled[index,]
test_ <- scaled[-index,]
y.train<-y[index,2]
y.test<-y[-index,2]

n <- names(train_)
f <- as.formula(paste("y.train ~", paste(n[!n %in% "y.train"], collapse = " + ")))
nn <- neuralnet(f,data=train_,hidden=c(9,6,4),linear.output=F)
#png("network.png", width=12, height=12, units="in", res=300)
#plot(nn,main='Neural network analysis',cex=1.2)
#grid(10, 10, lwd = 2)
#dev.off()
nn
pr.nn <- compute(nn,test_)
#pr.nn_ <- pr.nn$net.result*(max(datain$Class)-min(datain$Class))+min(datain$Class)
pr.nn_<-pr.nn$net.result
pr.nn_[pr.nn_<0.5]=0;pr.nn_[pr.nn_>=0.5]=1
#test.r <- (test_$Class)*(max(datain$Class)-min(datain$Class))+min(datain$Class)
table(pr.nn_,y.test)
mean(pr.nn_==y.test)

#corss-validation

set.seed(3)
cv.error <- NULL
index<-NULL
k <- 4
nn=list()
n<-length(y.train)

taille<-n%/%k
a<-runif(n)

ranka<-rank(a)
block<-(ranka-1)%/%taille+1
block<-as.factor(block)
table(block)


pbar <- create_progress_bar('text')
pbar$init(k)

for (i in 1:k){
  n <- names(train_)   
  f <- as.formula(paste("y.train[block!=i] ~", paste(n[!n %in% "y.train[block!=i,]"], collapse = " + ")))
  nn[[i]] <- neuralnet(f,data=train_[block!=i,],hidden=c(9,6,4),linear.output=F)
  pr.nn <- compute(nn[[i]],train_[block==i,1:13])
  pr.nn_<-pr.nn$net.result
  pr.nn_[pr.nn_<0.5]=0;pr.nn_[pr.nn_>=0.5]=1
  cv.error[i] <- 1-mean(pr.nn_==y.train[block==i])
  pbar$step()
}

cv.error
cv.error[which.min(cv.error)]

i=which.min(cv.error)
f <- as.formula(paste("y.train[block!=i] ~", paste(n[!n %in% "y.train[block!=i,]"], collapse = " + ")))
#nn <- neuralnet(f,data=train_[block!=i,],hidden=c(20,20,20,10,10,10,10),linear.output=F)
pr.nn <- compute(nn[[i]],test_[,1:13])
#pr.nn_ <- pr.nn$net.result*(max(datain$Class)-min(datain$Class))+min(datain$Class)
pr.nn_<-pr.nn$net.result
pr.nn_[pr.nn_<0.5]=0;pr.nn_[pr.nn_>=0.5]=1
#test.r <- (test_$Class)*(max(datain$Class)-min(datain$Class))+min(datain$Class)
table(pr.nn_,y.test)
mean(pr.nn_==y.test)

S=pr.nn$net.result;Y=y.test
roc.curve=function(s,print=FALSE){
  Ps=(S>s)*1
  FP=sum((Ps==1)*(Y==0))/sum(Y==0)
  TP=sum((Ps==1)*(Y==1))/sum(Y==1)
  if(print==TRUE){
    print(table(Observed=Y,Predicted=Ps))
  }
  vect=c(FP,TP)
  names(vect)=c("FPR","TPR")
  return(vect)
}
threshold = 0.5
roc.curve(threshold,print=TRUE)
mean(pr.nn_==y.test)
ROC.curve=Vectorize(roc.curve)

M.ROC=ROC.curve(seq(0,1,by=.01))
png("ROCn.png", width=9, height=7, units="in", res=300)
plot(M.ROC[1,],M.ROC[2,],col="black",lwd=4,type="l",ylab='TPR',xlab='FPR',main='ROC')
abline(0,1,col='red',lwd=4)
grid(10, 10, lwd = 2)
dev.off()

auc(M.ROC[1,],M.ROC[2,])

#SVM
set.seed(3)
index <- sample(1:nrow(tr),round(0.75*nrow(tr)))
train <- dat.log[index,]
test <- dat.log[-index,]
y.train<-y[index,-1]
y.test<-y[-index,-1]


datain.train<-cbind(train,y.train)
n <- names(datain.train)
f <- as.formula(paste("y.train ~", paste(n[!n %in% "y.train"], collapse = " + ")))
svm.fit=svm(f,data=datain.train,kernel ="radial",cost=1)
summary(svm.fit)
tune.out=tune(svm,f,data=datain.train, kernel ="radial"
              ,ranges = list( cost=c (0.01,0.1,1,10,100)))
bestmod=tune.out$best.model
svm.probs = predict(bestmod,test)
svm.pred=rep(0,22)
svm.pred[svm.probs>.5]=1
table(svm.pred,y.test)
mean(svm.pred==y.test)

S=svm.probs;Y=y.test
roc.curve=function(s,print=FALSE){
  Ps=(S>s)*1
  FP=sum((Ps==1)*(Y==0))/sum(Y==0)
  TP=sum((Ps==1)*(Y==1))/sum(Y==1)
  if(print==TRUE){
    print(table(Observed=Y,Predicted=Ps))
  }
  vect=c(FP,TP)
  names(vect)=c("FPR","TPR")
  return(vect)
}
threshold = 0.5
roc.curve(threshold,print=TRUE)
mean(svm.pred==y.test)
ROC.curve=Vectorize(roc.curve)

M.ROC=ROC.curve(seq(0,1,by=.01))
png("ROCs.png", width=9, height=7, units="in", res=300)
plot(M.ROC[1,],M.ROC[2,],col="black",lwd=4,type="l",ylab='TPR',xlab='FPR',main='ROC',xlim=c(0,1),ylim=c(0,1))
abline(0,1,col='red',lwd=4)
grid(10, 10, lwd = 2)
dev.off()

auc(M.ROC[1,],M.ROC[2,])