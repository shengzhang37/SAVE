library(XML)


data<-xmlParse("559-ws-training.xml")
rootnode=xmlRoot(data)
variables=xmlChildren(rootnode)
nvar=xmlSize(rootnode)
person1=list()
varnames=rep("",nvar)
for (i in 1:nvar){
  vari=variables[[i]]
  varnames[i]=xmlName(variables[[i]])
  data_var<-t(as.matrix(xmlSApply(vari,xmlAttrs)))
  person1[[i]]=data_var
}

data<-xmlParse("563-ws-training.xml")
rootnode=xmlRoot(data)
variables=xmlChildren(rootnode)
nvar=xmlSize(rootnode)
person2=list()
varnames=rep("",nvar)
for (i in 1:nvar){
  vari=variables[[i]]
  varnames[i]=xmlName(variables[[i]])
  data_var<-t(as.matrix(xmlSApply(vari,xmlAttrs)))
  person2[[i]]=data_var
}


data<-xmlParse("570-ws-training.xml")
rootnode=xmlRoot(data)
variables=xmlChildren(rootnode)
nvar=xmlSize(rootnode)
person3=list()
varnames=rep("",nvar)
for (i in 1:nvar){
  vari=variables[[i]]
  varnames[i]=xmlName(variables[[i]])
  data_var<-t(as.matrix(xmlSApply(vari,xmlAttrs)))
  person3[[i]]=data_var
}

data<-xmlParse("575-ws-training.xml")
rootnode=xmlRoot(data)
variables=xmlChildren(rootnode)
nvar=xmlSize(rootnode)
person4=list()
varnames=rep("",nvar)
for (i in 1:nvar){
  vari=variables[[i]]
  varnames[i]=xmlName(variables[[i]])
  data_var<-t(as.matrix(xmlSApply(vari,xmlAttrs)))
  person4[[i]]=data_var
}

data<-xmlParse("588-ws-training.xml")
rootnode=xmlRoot(data)
variables=xmlChildren(rootnode)
nvar=xmlSize(rootnode)
person5=list()
varnames=rep("",nvar)
for (i in 1:nvar){
  vari=variables[[i]]
  varnames[i]=xmlName(variables[[i]])
  data_var<-t(as.matrix(xmlSApply(vari,xmlAttrs)))
  person5[[i]]=data_var
}

data<-xmlParse("591-ws-training.xml")
rootnode=xmlRoot(data)
variables=xmlChildren(rootnode)
nvar=xmlSize(rootnode)
person6=list()
varnames=rep("",nvar)
for (i in 1:nvar){
  vari=variables[[i]]
  varnames[i]=xmlName(variables[[i]])
  data_var<-t(as.matrix(xmlSApply(vari,xmlAttrs)))
  person6[[i]]=data_var
}

save.image("data-train.Rdata")

















######################
ICG=function(x,a=1.35,b=2){
  return(-(x<80)*((abs(80-x))^b)/30-(x>140)*((abs(x-140))^a)/30)
}

HyperIndex=function(x,a=1.35){
  return(-(x>140)*((abs(x-140))^a)/30)
}

HypoIndex=function(x,b=2){
  return(-(x<80)*((abs(80-x))^b)/30)
}
M100=function(x){
  return(-1000*(abs(log10(x/100)))^3)
}

load("data-train.Rdata")
library(lubridate)
person1_use=list(glucose=person1[[1]],basal=person1[[3]], tempbasal=person1[[4]], bolus=person1[[5]][,c(1,4)],meal=person1[[6]],heart=person1[[13]],gsr=person1[[14]],skintemp=person1[[15]],airtemp=person1[[16]],steps=person1[[17]])
person2_use=list(glucose=person2[[1]],basal=person2[[3]], tempbasal=person2[[4]],bolus=person2[[5]][,c(1,4)],meal=person2[[6]],heart=person2[[13]],gsr=person2[[14]],skintemp=person2[[15]],airtemp=person2[[16]],steps=person2[[17]])
person3_use=list(glucose=person3[[1]],basal=person3[[3]], tempbasal=person3[[4]],bolus=person3[[5]][,c(1,4)],meal=person3[[6]],heart=person3[[13]],gsr=person3[[14]],skintemp=person3[[15]],airtemp=person3[[16]],steps=person3[[17]])
person4_use=list(glucose=person4[[1]],basal=person4[[3]], tempbasal=person4[[4]],bolus=person4[[5]][,c(1,4)],meal=person4[[6]],heart=person4[[13]],gsr=person4[[14]],skintemp=person4[[15]],airtemp=person4[[16]],steps=person4[[17]])
person5_use=list(glucose=person5[[1]],basal=person5[[3]], tempbasal=person5[[4]],bolus=person5[[5]][,c(1,4)],meal=person5[[6]],heart=person5[[13]],gsr=person5[[14]],skintemp=person5[[15]],airtemp=person5[[16]],steps=person5[[17]])
person6_use=list(glucose=person6[[1]],basal=person6[[3]], tempbasal=person6[[4]],bolus=person6[[5]][,c(1,4)],meal=person6[[6]],heart=person6[[13]],gsr=person6[[14]],skintemp=person6[[15]],airtemp=person6[[16]],steps=person6[[17]])



halfhour=function(x){
  return(2*hour(x)+floor(minute(x)/30))
}
halfminute=function(x){
  return(minute(x)%%30)
}
daytime=function(x){
  return(60*hour(x)+minute(x))
}
#data transforming for person1
#deal with date

person1_5min=list(glucose=data.frame( day=as.numeric(date(dmy_hms(person1_use$glucose[,1]))-date(dmy_hms(person1_use$glucose[1,1]))), 
                                      fiveminute=floor(daytime(dmy_hms(person1_use$glucose[,1]))/5),
                                      glucose=as.numeric(person1_use$glucose[,2])) ,
                  basal=data.frame(day=as.numeric(date(dmy_hms(person1_use$basal[,1]))-date(dmy_hms(person1_use$glucose[1,1]))), 
                                   daytime=daytime(dmy_hms(person1_use$basal[,1])),
                                   rate=as.numeric(person1_use$basal[,2])),
                  tempbasal=data.frame(start_day=as.numeric(date(dmy_hms(person1_use$tempbasal[,1]))-date(dmy_hms(person1_use$glucose[1,1]))),
                                       start_daytime=daytime(dmy_hms(person1_use$tempbasal[,1])),
                                       end_day=as.numeric(date(dmy_hms(person1_use$tempbasal[,2]))-date(dmy_hms(person1_use$glucose[1,1]))),
                                       end_daytime=daytime(dmy_hms(person1_use$tempbasal[,2])),
                                       rate=as.numeric(person1_use$tempbasal[,3])  ),
                  bolus=data.frame(day=as.numeric(date(dmy_hms(person1_use$bolus[,1]))-date(dmy_hms(person1_use$glucose[1,1]))), 
                                   fiveminute=floor(daytime(dmy_hms(person1_use$bolus[,1]))/5),
                                   dose=as.numeric(person1_use$bolus[,2])), 
                  meal=data.frame(day=as.numeric(date(dmy_hms(person1_use$meal[,1]))-date(dmy_hms(person1_use$glucose[1,1]))), 
                                  daytime=daytime(dmy_hms(person1_use$meal[,1])),
                                  carb=as.numeric(person1_use$meal[,3])),
                  heart=data.frame(day=as.numeric(date(dmy_hms(person1_use$heart[,1]))-date(dmy_hms(person1_use$glucose[1,1]))), 
                                   fiveminute=floor(daytime(dmy_hms(person1_use$heart[,1]))/5),
                                   heart=as.numeric(person1_use$heart[,2])),
                  gsr=data.frame(day=as.numeric(date(dmy_hms(person1_use$gsr[,1]))-date(dmy_hms(person1_use$glucose[1,1]))),
                                 fiveminute=floor(daytime(dmy_hms(person1_use$gsr[,1]))/5),
                                 gsr=as.numeric(person1_use$gsr[,2])),
                  skintemp=data.frame(day=as.numeric(date(dmy_hms(person1_use$skintemp[,1]))-date(dmy_hms(person1_use$glucose[1,1]))),
                                      fiveminute=floor(daytime(dmy_hms(person1_use$skintemp[,1]))/5),
                                      skintemp=as.numeric(person1_use$skintemp[,2])),
                  airtemp=data.frame(day=as.numeric(date(dmy_hms(person1_use$airtemp[,1]))-date(dmy_hms(person1_use$glucose[1,1]))),
                                     fiveminute=floor(daytime(dmy_hms(person1_use$airtemp[,1]))/5),
                                     airtemp=as.numeric(person1_use$airtemp[,2])),
                  steps=data.frame(day=as.numeric(date(dmy_hms(person1_use$steps[,1]))-date(dmy_hms(person1_use$glucose[1,1]))),
                                   fiveminute=floor(daytime(dmy_hms(person1_use$steps[,1]))/5),
                                   steps=as.numeric(person1_use$steps[,2]))
                  )

#deal with basal
nday=max(person1_5min$glucose$day)-min(person1_5min$glucose$day)+1
#deal with basal
date5min=data.frame(day=rep(seq(0,nday-1),each=288),fiveminute=rep(seq(0,287),nday))

tbasal=date5min
tbasal$rate=NA
ob=1
trate=0
person1_5min$basal=rbind(person1_5min$basal, c(max(person1_5min$glucose$day)+1,0,0))
for (i in 1:(288*nday)){
  if ((person1_5min$basal$day[ob]-min(person1_5min$glucose$day))*288+floor(person1_5min$basal$daytime[ob]/5)>i-1) {
    tbasal$rate[i]=trate
  } else{
    begin5minute=0
    tbasal$rate[i]=0
    while ((person1_5min$basal$day[ob]-min(person1_5min$glucose$day))*288+floor(person1_5min$basal$daytime[ob]/5)<=i-1) {
      tbasal$rate[i]=tbasal$rate[i]+ trate* (person1_5min$basal$daytime[ob]%%5-begin5minute)/5
      begin5minute=person1_5min$basal$daytime[ob]%%5
      trate=person1_5min$basal$rate[ob]
      ob=ob+1
    }
    tbasal$rate[i]=tbasal$rate[i]+ trate* (5-begin5minute)/5
  }
}
#adjust temp basal
for (i in 1:nrow(person1_5min$tempbasal)){
  begin5min_whole=(person1_5min$tempbasal$start_day[i]-min(person1_5min$glucose$day))*288+floor(person1_5min$tempbasal$start_daytime[i]/5)+1
  end5min_whole=(person1_5min$tempbasal$end_day[i]-min(person1_5min$glucose$day))*288+floor(person1_5min$tempbasal$end_daytime[i]/5)+1
  begin5min_rest=person1_5min$tempbasal$start_daytime[i]%%5
  end5min_rest=person1_5min$tempbasal$end_daytime[i]%%5
  trate=person1_5min$tempbasal$rate[i]
  while (begin5min_whole<end5min_whole){
    tbasal[begin5min_whole,]$rate=tbasal[begin5min_whole,]$rate*(begin5min_rest)/5+trate*(5-begin5min_rest)/5
    begin5min_whole=begin5min_whole+1
    begin5min_rest=0
  }
  tbasal[ begin5min_whole,]$rate=tbasal[ begin5min_whole,]$rate*(begin5min_rest+5-end5min_rest)/5 +trate*(end5min_rest-begin5min_rest)/5
}


#deal with bolus
tbolus=aggregate(dose~day+fiveminute, data=person1_5min$bolus,FUN=sum,na.rm=TRUE)
tbolus=merge(date5min,tbolus,by=c("day","fiveminute"),all.x=TRUE)
tbolus$lastmeal_day=NA
tbolus$lastmeal_daytime=NA
tbolus$lastmeal_carb=NA
tbolus$lastmeal_timediff=NA
tbolus$nextmeal_day=NA
tbolus$nextmeal_daytime=NA
tbolus$nextmeal_carb=NA
tbolus$nextmeal_timediff=NA

#add meal to the bolus dataset
lastmeal_day=NA
lastmeal_daytime=NA
lastmeal_carb=NA
nextmeal_day=person1_5min$meal$day[1]
nextmeal_daytime=person1_5min$meal$daytime[1]
nextmeal_carb=person1_5min$meal$carb[1]

person1_5min$meal=rbind(person1_5min$meal,c(max(person1_5min$glucose$day)+1,0,NA))

ob=1
for (day in min(person1_5min$glucose$day):max(person1_5min$glucose$day)){
  for (fiveminute in 0:287){
    i=day*288+fiveminute
    if ((day*1440+fiveminute*5)> (person1_5min$meal$day[ob]*1440+person1_5min$meal$daytime[ob])){
      lastmeal_day=person1_5min$meal$day[ob]
      lastmeal_daytime=person1_5min$meal$daytime[ob]
      lastmeal_carb=person1_5min$meal$carb[ob]
      ob=ob+1
    } 
    tbolus$lastmeal_day[i+1]=lastmeal_day
    tbolus$lastmeal_daytime[i+1]=lastmeal_daytime
    tbolus$lastmeal_carb[i+1]=lastmeal_carb
    tbolus$lastmeal_timediff[i+1]=(day*1440+fiveminute*5)-(lastmeal_day*1440+lastmeal_daytime)
    tbolus$nextmeal_day[i+1]=person1_5min$meal$day[ob]
    tbolus$nextmeal_daytime[i+1]=person1_5min$meal$daytime[ob]
    tbolus$nextmeal_carb[i+1]=person1_5min$meal$carb[ob]
    tbolus$nextmeal_timediff[i+1]=(person1_5min$meal$day[ob]*1440+person1_5min$meal$daytime[ob])-(day*1440+fiveminute*5) 
  }
}

#glucose
tglucose=aggregate(glucose~day+fiveminute,data=person1_5min$glucose,FUN=mean)
names(tglucose)=c("day","fiveminute","glucose")
tglucose=merge(date5min,tglucose,by=c("day","fiveminute"),all.x=TRUE)
tglucose$stability=ICG(tglucose$glucose)
tglucose$hyper=HyperIndex(tglucose$glucose)
tglucose$hypo=HypoIndex(tglucose$glucose)

theart=aggregate(heart~day+fiveminute, data=person1_5min$heart,FUN=mean)
tgsr=aggregate(gsr~day+fiveminute,data=person1_5min$gsr,FUN=mean)
tskintemp=aggregate(skintemp~day+fiveminute,data=person1_5min$skintemp,FUN=mean)
tairtemp=aggregate(airtemp~day+fiveminute,data=person1_5min$airtemp,FUN=mean)
tsteps=aggregate(steps~day+fiveminute,data=person1_5min$steps,FUN=mean)

person1_final=merge(tbasal,tbolus,by=c("day","fiveminute"),all=TRUE)
person1_final=merge(person1_final,tglucose,by=c("day","fiveminute"),all=TRUE)
person1_final=merge(person1_final,theart,by=c("day","fiveminute"),all=TRUE)
person1_final=merge(person1_final,tgsr,by=c("day","fiveminute"),all=TRUE)
person1_final=merge(person1_final,tskintemp,by=c("day","fiveminute"),all=TRUE)
person1_final=merge(person1_final,tairtemp,by=c("day","fiveminute"),all=TRUE)
person1_final=merge(person1_final,tsteps,by=c("day","fiveminute"),all=TRUE)



person1_final=person1_final[order(person1_final$day,person1_final$fiveminute),]
person1_final$dose[is.na(person1_final$dose)]=0

#delete observations that are missing too much

write.csv(person1_final, "fiveminute/person1-5min-train.csv",row.names=FALSE)
########################
########################
#######################
#person2################
#######################

person2_5min=list(glucose=data.frame( day=as.numeric(date(dmy_hms(person2_use$glucose[,1]))-date(dmy_hms(person2_use$glucose[1,1]))), 
                                      fiveminute=floor(daytime(dmy_hms(person2_use$glucose[,1]))/5),
                                      glucose=as.numeric(person2_use$glucose[,2])) ,
                  basal=data.frame(day=as.numeric(date(dmy_hms(person2_use$basal[,1]))-date(dmy_hms(person2_use$glucose[1,1]))), 
                                   daytime=daytime(dmy_hms(person2_use$basal[,1])),
                                   rate=as.numeric(person2_use$basal[,2])),
                  tempbasal=data.frame(start_day=as.numeric(date(dmy_hms(person2_use$tempbasal[,1]))-date(dmy_hms(person2_use$glucose[1,1]))),
                                       start_daytime=daytime(dmy_hms(person2_use$tempbasal[,1])),
                                       end_day=as.numeric(date(dmy_hms(person2_use$tempbasal[,2]))-date(dmy_hms(person2_use$glucose[1,1]))),
                                       end_daytime=daytime(dmy_hms(person2_use$tempbasal[,2])),
                                       rate=as.numeric(person2_use$tempbasal[,3])  ),
                  bolus=data.frame(day=as.numeric(date(dmy_hms(person2_use$bolus[,1]))-date(dmy_hms(person2_use$glucose[1,1]))), 
                                   fiveminute=floor(daytime(dmy_hms(person2_use$bolus[,1]))/5),
                                   dose=as.numeric(person2_use$bolus[,2])), 
                  meal=data.frame(day=as.numeric(date(dmy_hms(person2_use$meal[,1]))-date(dmy_hms(person2_use$glucose[1,1]))), 
                                  daytime=daytime(dmy_hms(person2_use$meal[,1])),
                                  carb=as.numeric(person2_use$meal[,3])),
                  heart=data.frame(day=as.numeric(date(dmy_hms(person2_use$heart[,1]))-date(dmy_hms(person2_use$glucose[1,1]))), 
                                   fiveminute=floor(daytime(dmy_hms(person2_use$heart[,1]))/5),
                                   heart=as.numeric(person2_use$heart[,2])),
                  gsr=data.frame(day=as.numeric(date(dmy_hms(person2_use$gsr[,1]))-date(dmy_hms(person2_use$glucose[1,1]))),
                                 fiveminute=floor(daytime(dmy_hms(person2_use$gsr[,1]))/5),
                                 gsr=as.numeric(person2_use$gsr[,2])),
                  skintemp=data.frame(day=as.numeric(date(dmy_hms(person2_use$skintemp[,1]))-date(dmy_hms(person2_use$glucose[1,1]))),
                                      fiveminute=floor(daytime(dmy_hms(person2_use$skintemp[,1]))/5),
                                      skintemp=as.numeric(person2_use$skintemp[,2])),
                  airtemp=data.frame(day=as.numeric(date(dmy_hms(person2_use$airtemp[,1]))-date(dmy_hms(person2_use$glucose[1,1]))),
                                     fiveminute=floor(daytime(dmy_hms(person2_use$airtemp[,1]))/5),
                                     airtemp=as.numeric(person2_use$airtemp[,2])),
                  steps=data.frame(day=as.numeric(date(dmy_hms(person2_use$steps[,1]))-date(dmy_hms(person2_use$glucose[1,1]))),
                                   fiveminute=floor(daytime(dmy_hms(person2_use$steps[,1]))/5),
                                   steps=as.numeric(person2_use$steps[,2]))
)

#deal with basal
nday=max(person2_5min$glucose$day)-min(person2_5min$glucose$day)+1
#deal with basal
date5min=data.frame(day=rep(seq(0,nday-1),each=288),fiveminute=rep(seq(0,287),nday))

tbasal=date5min
tbasal$rate=NA
ob=1
trate=0
person2_5min$basal=rbind(person2_5min$basal, c(max(person2_5min$glucose$day)+1,0,0))
for (i in 1:(288*nday)){
  if ((person2_5min$basal$day[ob]-min(person2_5min$glucose$day))*288+floor(person2_5min$basal$daytime[ob]/5)>i-1) {
    tbasal$rate[i]=trate
  } else{
    begin5minute=0
    tbasal$rate[i]=0
    while ((person2_5min$basal$day[ob]-min(person2_5min$glucose$day))*288+floor(person2_5min$basal$daytime[ob]/5)<=i-1) {
      tbasal$rate[i]=tbasal$rate[i]+ trate* (person2_5min$basal$daytime[ob]%%5-begin5minute)/5
      begin5minute=person2_5min$basal$daytime[ob]%%5
      trate=person2_5min$basal$rate[ob]
      ob=ob+1
    }
    tbasal$rate[i]=tbasal$rate[i]+ trate* (5-begin5minute)/5
  }
}
#adjust temp basal
for (i in 1:nrow(person2_5min$tempbasal)){
  begin5min_whole=(person2_5min$tempbasal$start_day[i]-min(person2_5min$glucose$day))*288+floor(person2_5min$tempbasal$start_daytime[i]/5)+1
  end5min_whole=(person2_5min$tempbasal$end_day[i]-min(person2_5min$glucose$day))*288+floor(person2_5min$tempbasal$end_daytime[i]/5)+1
  begin5min_rest=person2_5min$tempbasal$start_daytime[i]%%5
  end5min_rest=person2_5min$tempbasal$end_daytime[i]%%5
  trate=person2_5min$tempbasal$rate[i]
  while (begin5min_whole<end5min_whole){
    tbasal[begin5min_whole,]$rate=tbasal[begin5min_whole,]$rate*(begin5min_rest)/5+trate*(5-begin5min_rest)/5
    begin5min_whole=begin5min_whole+1
    begin5min_rest=0
  }
  tbasal[ begin5min_whole,]$rate=tbasal[ begin5min_whole,]$rate*(begin5min_rest+5-end5min_rest)/5 +trate*(end5min_rest-begin5min_rest)/5
}


#deal with bolus
tbolus=aggregate(dose~day+fiveminute, data=person2_5min$bolus,FUN=sum,na.rm=TRUE)
tbolus=merge(date5min,tbolus,by=c("day","fiveminute"),all.x=TRUE)
tbolus$lastmeal_day=NA
tbolus$lastmeal_daytime=NA
tbolus$lastmeal_carb=NA
tbolus$lastmeal_timediff=NA
tbolus$nextmeal_day=NA
tbolus$nextmeal_daytime=NA
tbolus$nextmeal_carb=NA
tbolus$nextmeal_timediff=NA

#add meal to the bolus dataset
lastmeal_day=NA
lastmeal_daytime=NA
lastmeal_carb=NA
nextmeal_day=person2_5min$meal$day[1]
nextmeal_daytime=person2_5min$meal$daytime[1]
nextmeal_carb=person2_5min$meal$carb[1]

person2_5min$meal=rbind(person2_5min$meal,c(max(person2_5min$glucose$day)+1,0,NA))

ob=1
for (day in min(person2_5min$glucose$day):max(person2_5min$glucose$day)){
  for (fiveminute in 0:287){
    i=day*288+fiveminute
    if ((day*1440+fiveminute*5)> (person2_5min$meal$day[ob]*1440+person2_5min$meal$daytime[ob])){
      lastmeal_day=person2_5min$meal$day[ob]
      lastmeal_daytime=person2_5min$meal$daytime[ob]
      lastmeal_carb=person2_5min$meal$carb[ob]
      ob=ob+1
    } 
    tbolus$lastmeal_day[i+1]=lastmeal_day
    tbolus$lastmeal_daytime[i+1]=lastmeal_daytime
    tbolus$lastmeal_carb[i+1]=lastmeal_carb
    tbolus$lastmeal_timediff[i+1]=(day*1440+fiveminute*5)-(lastmeal_day*1440+lastmeal_daytime)
    tbolus$nextmeal_day[i+1]=person2_5min$meal$day[ob]
    tbolus$nextmeal_daytime[i+1]=person2_5min$meal$daytime[ob]
    tbolus$nextmeal_carb[i+1]=person2_5min$meal$carb[ob]
    tbolus$nextmeal_timediff[i+1]=(person2_5min$meal$day[ob]*1440+person2_5min$meal$daytime[ob])-(day*1440+fiveminute*5) 
  }
}

#glucose
tglucose=aggregate(glucose~day+fiveminute,data=person2_5min$glucose,FUN=mean)
names(tglucose)=c("day","fiveminute","glucose")
tglucose=merge(date5min,tglucose,by=c("day","fiveminute"),all.x=TRUE)
tglucose$stability=ICG(tglucose$glucose)
tglucose$hyper=HyperIndex(tglucose$glucose)
tglucose$hypo=HypoIndex(tglucose$glucose)

theart=aggregate(heart~day+fiveminute, data=person2_5min$heart,FUN=mean)
tgsr=aggregate(gsr~day+fiveminute,data=person2_5min$gsr,FUN=mean)
tskintemp=aggregate(skintemp~day+fiveminute,data=person2_5min$skintemp,FUN=mean)
tairtemp=aggregate(airtemp~day+fiveminute,data=person2_5min$airtemp,FUN=mean)
tsteps=aggregate(steps~day+fiveminute,data=person2_5min$steps,FUN=mean)

person2_final=merge(tbasal,tbolus,by=c("day","fiveminute"),all=TRUE)
person2_final=merge(person2_final,tglucose,by=c("day","fiveminute"),all=TRUE)
person2_final=merge(person2_final,theart,by=c("day","fiveminute"),all=TRUE)
person2_final=merge(person2_final,tgsr,by=c("day","fiveminute"),all=TRUE)
person2_final=merge(person2_final,tskintemp,by=c("day","fiveminute"),all=TRUE)
person2_final=merge(person2_final,tairtemp,by=c("day","fiveminute"),all=TRUE)
person2_final=merge(person2_final,tsteps,by=c("day","fiveminute"),all=TRUE)



person2_final=person2_final[order(person2_final$day,person2_final$fiveminute),]
person2_final$dose[is.na(person2_final$dose)]=0

#delete observations that are missing too much

write.csv(person2_final, "fiveminute/person2-5min-train.csv",row.names=FALSE)


########################
########################
#######################
#person3################
#######################
person3_5min=list(glucose=data.frame( day=as.numeric(date(dmy_hms(person3_use$glucose[,1]))-date(dmy_hms(person3_use$glucose[1,1]))), 
                                      fiveminute=floor(daytime(dmy_hms(person3_use$glucose[,1]))/5),
                                      glucose=as.numeric(person3_use$glucose[,2])) ,
                  basal=data.frame(day=as.numeric(date(dmy_hms(person3_use$basal[,1]))-date(dmy_hms(person3_use$glucose[1,1]))), 
                                   daytime=daytime(dmy_hms(person3_use$basal[,1])),
                                   rate=as.numeric(person3_use$basal[,2])),
                  tempbasal=data.frame(start_day=as.numeric(date(dmy_hms(person3_use$tempbasal[,1]))-date(dmy_hms(person3_use$glucose[1,1]))),
                                       start_daytime=daytime(dmy_hms(person3_use$tempbasal[,1])),
                                       end_day=as.numeric(date(dmy_hms(person3_use$tempbasal[,2]))-date(dmy_hms(person3_use$glucose[1,1]))),
                                       end_daytime=daytime(dmy_hms(person3_use$tempbasal[,2])),
                                       rate=as.numeric(person3_use$tempbasal[,3])  ),
                  bolus=data.frame(day=as.numeric(date(dmy_hms(person3_use$bolus[,1]))-date(dmy_hms(person3_use$glucose[1,1]))), 
                                   fiveminute=floor(daytime(dmy_hms(person3_use$bolus[,1]))/5),
                                   dose=as.numeric(person3_use$bolus[,2])), 
                  meal=data.frame(day=as.numeric(date(dmy_hms(person3_use$meal[,1]))-date(dmy_hms(person3_use$glucose[1,1]))), 
                                  daytime=daytime(dmy_hms(person3_use$meal[,1])),
                                  carb=as.numeric(person3_use$meal[,3])),
                  heart=data.frame(day=as.numeric(date(dmy_hms(person3_use$heart[,1]))-date(dmy_hms(person3_use$glucose[1,1]))), 
                                   fiveminute=floor(daytime(dmy_hms(person3_use$heart[,1]))/5),
                                   heart=as.numeric(person3_use$heart[,2])),
                  gsr=data.frame(day=as.numeric(date(dmy_hms(person3_use$gsr[,1]))-date(dmy_hms(person3_use$glucose[1,1]))),
                                 fiveminute=floor(daytime(dmy_hms(person3_use$gsr[,1]))/5),
                                 gsr=as.numeric(person3_use$gsr[,2])),
                  skintemp=data.frame(day=as.numeric(date(dmy_hms(person3_use$skintemp[,1]))-date(dmy_hms(person3_use$glucose[1,1]))),
                                      fiveminute=floor(daytime(dmy_hms(person3_use$skintemp[,1]))/5),
                                      skintemp=as.numeric(person3_use$skintemp[,2])),
                  airtemp=data.frame(day=as.numeric(date(dmy_hms(person3_use$airtemp[,1]))-date(dmy_hms(person3_use$glucose[1,1]))),
                                     fiveminute=floor(daytime(dmy_hms(person3_use$airtemp[,1]))/5),
                                     airtemp=as.numeric(person3_use$airtemp[,2])),
                  steps=data.frame(day=as.numeric(date(dmy_hms(person3_use$steps[,1]))-date(dmy_hms(person3_use$glucose[1,1]))),
                                   fiveminute=floor(daytime(dmy_hms(person3_use$steps[,1]))/5),
                                   steps=as.numeric(person3_use$steps[,2]))
)

#deal with basal
nday=max(person3_5min$glucose$day)-min(person3_5min$glucose$day)+1
#deal with basal
date5min=data.frame(day=rep(seq(0,nday-1),each=288),fiveminute=rep(seq(0,287),nday))

tbasal=date5min
tbasal$rate=NA
ob=1
trate=0
person3_5min$basal=rbind(person3_5min$basal, c(max(person3_5min$glucose$day)+1,0,0))
for (i in 1:(288*nday)){
  if ((person3_5min$basal$day[ob]-min(person3_5min$glucose$day))*288+floor(person3_5min$basal$daytime[ob]/5)>i-1) {
    tbasal$rate[i]=trate
  } else{
    begin5minute=0
    tbasal$rate[i]=0
    while ((person3_5min$basal$day[ob]-min(person3_5min$glucose$day))*288+floor(person3_5min$basal$daytime[ob]/5)<=i-1) {
      tbasal$rate[i]=tbasal$rate[i]+ trate* (person3_5min$basal$daytime[ob]%%5-begin5minute)/5
      begin5minute=person3_5min$basal$daytime[ob]%%5
      trate=person3_5min$basal$rate[ob]
      ob=ob+1
    }
    tbasal$rate[i]=tbasal$rate[i]+ trate* (5-begin5minute)/5
  }
}
#adjust temp basal
for (i in 1:nrow(person3_5min$tempbasal)){
  begin5min_whole=(person3_5min$tempbasal$start_day[i]-min(person3_5min$glucose$day))*288+floor(person3_5min$tempbasal$start_daytime[i]/5)+1
  end5min_whole=(person3_5min$tempbasal$end_day[i]-min(person3_5min$glucose$day))*288+floor(person3_5min$tempbasal$end_daytime[i]/5)+1
  begin5min_rest=person3_5min$tempbasal$start_daytime[i]%%5
  end5min_rest=person3_5min$tempbasal$end_daytime[i]%%5
  trate=person3_5min$tempbasal$rate[i]
  while (begin5min_whole<end5min_whole){
    tbasal[begin5min_whole,]$rate=tbasal[begin5min_whole,]$rate*(begin5min_rest)/5+trate*(5-begin5min_rest)/5
    begin5min_whole=begin5min_whole+1
    begin5min_rest=0
  }
  tbasal[ begin5min_whole,]$rate=tbasal[ begin5min_whole,]$rate*(begin5min_rest+5-end5min_rest)/5 +trate*(end5min_rest-begin5min_rest)/5
}


#deal with bolus
tbolus=aggregate(dose~day+fiveminute, data=person3_5min$bolus,FUN=sum,na.rm=TRUE)
tbolus=merge(date5min,tbolus,by=c("day","fiveminute"),all.x=TRUE)
tbolus$lastmeal_day=NA
tbolus$lastmeal_daytime=NA
tbolus$lastmeal_carb=NA
tbolus$lastmeal_timediff=NA
tbolus$nextmeal_day=NA
tbolus$nextmeal_daytime=NA
tbolus$nextmeal_carb=NA
tbolus$nextmeal_timediff=NA

#add meal to the bolus dataset
lastmeal_day=NA
lastmeal_daytime=NA
lastmeal_carb=NA
nextmeal_day=person3_5min$meal$day[1]
nextmeal_daytime=person3_5min$meal$daytime[1]
nextmeal_carb=person3_5min$meal$carb[1]

person3_5min$meal=rbind(person3_5min$meal,c(max(person3_5min$glucose$day)+1,0,NA))

ob=1
for (day in min(person3_5min$glucose$day):max(person3_5min$glucose$day)){
  for (fiveminute in 0:287){
    i=day*288+fiveminute
    if ((day*1440+fiveminute*5)> (person3_5min$meal$day[ob]*1440+person3_5min$meal$daytime[ob])){
      lastmeal_day=person3_5min$meal$day[ob]
      lastmeal_daytime=person3_5min$meal$daytime[ob]
      lastmeal_carb=person3_5min$meal$carb[ob]
      ob=ob+1
    } 
    tbolus$lastmeal_day[i+1]=lastmeal_day
    tbolus$lastmeal_daytime[i+1]=lastmeal_daytime
    tbolus$lastmeal_carb[i+1]=lastmeal_carb
    tbolus$lastmeal_timediff[i+1]=(day*1440+fiveminute*5)-(lastmeal_day*1440+lastmeal_daytime)
    tbolus$nextmeal_day[i+1]=person3_5min$meal$day[ob]
    tbolus$nextmeal_daytime[i+1]=person3_5min$meal$daytime[ob]
    tbolus$nextmeal_carb[i+1]=person3_5min$meal$carb[ob]
    tbolus$nextmeal_timediff[i+1]=(person3_5min$meal$day[ob]*1440+person3_5min$meal$daytime[ob])-(day*1440+fiveminute*5) 
  }
}

#glucose
tglucose=aggregate(glucose~day+fiveminute,data=person3_5min$glucose,FUN=mean)
names(tglucose)=c("day","fiveminute","glucose")
tglucose=merge(date5min,tglucose,by=c("day","fiveminute"),all.x=TRUE)
tglucose$stability=ICG(tglucose$glucose)
tglucose$hyper=HyperIndex(tglucose$glucose)
tglucose$hypo=HypoIndex(tglucose$glucose)

theart=aggregate(heart~day+fiveminute, data=person3_5min$heart,FUN=mean)
tgsr=aggregate(gsr~day+fiveminute,data=person3_5min$gsr,FUN=mean)
tskintemp=aggregate(skintemp~day+fiveminute,data=person3_5min$skintemp,FUN=mean)
tairtemp=aggregate(airtemp~day+fiveminute,data=person3_5min$airtemp,FUN=mean)
tsteps=aggregate(steps~day+fiveminute,data=person3_5min$steps,FUN=mean)

person3_final=merge(tbasal,tbolus,by=c("day","fiveminute"),all=TRUE)
person3_final=merge(person3_final,tglucose,by=c("day","fiveminute"),all=TRUE)
person3_final=merge(person3_final,theart,by=c("day","fiveminute"),all=TRUE)
person3_final=merge(person3_final,tgsr,by=c("day","fiveminute"),all=TRUE)
person3_final=merge(person3_final,tskintemp,by=c("day","fiveminute"),all=TRUE)
person3_final=merge(person3_final,tairtemp,by=c("day","fiveminute"),all=TRUE)
person3_final=merge(person3_final,tsteps,by=c("day","fiveminute"),all=TRUE)



person3_final=person3_final[order(person3_final$day,person3_final$fiveminute),]
person3_final$dose[is.na(person3_final$dose)]=0

#delete observations that are missing too much

write.csv(person3_final, "fiveminute/person3-5min-train.csv",row.names=FALSE)



########################
########################
#######################
#person4################
#######################
person4_5min=list(glucose=data.frame( day=as.numeric(date(dmy_hms(person4_use$glucose[,1]))-date(dmy_hms(person4_use$glucose[1,1]))), 
                                      fiveminute=floor(daytime(dmy_hms(person4_use$glucose[,1]))/5),
                                      glucose=as.numeric(person4_use$glucose[,2])) ,
                  basal=data.frame(day=as.numeric(date(dmy_hms(person4_use$basal[,1]))-date(dmy_hms(person4_use$glucose[1,1]))), 
                                   daytime=daytime(dmy_hms(person4_use$basal[,1])),
                                   rate=as.numeric(person4_use$basal[,2])),
                  tempbasal=data.frame(start_day=as.numeric(date(dmy_hms(person4_use$tempbasal[,1]))-date(dmy_hms(person4_use$glucose[1,1]))),
                                       start_daytime=daytime(dmy_hms(person4_use$tempbasal[,1])),
                                       end_day=as.numeric(date(dmy_hms(person4_use$tempbasal[,2]))-date(dmy_hms(person4_use$glucose[1,1]))),
                                       end_daytime=daytime(dmy_hms(person4_use$tempbasal[,2])),
                                       rate=as.numeric(person4_use$tempbasal[,3])  ),
                  bolus=data.frame(day=as.numeric(date(dmy_hms(person4_use$bolus[,1]))-date(dmy_hms(person4_use$glucose[1,1]))), 
                                   fiveminute=floor(daytime(dmy_hms(person4_use$bolus[,1]))/5),
                                   dose=as.numeric(person4_use$bolus[,2])), 
                  meal=data.frame(day=as.numeric(date(dmy_hms(person4_use$meal[,1]))-date(dmy_hms(person4_use$glucose[1,1]))), 
                                  daytime=daytime(dmy_hms(person4_use$meal[,1])),
                                  carb=as.numeric(person4_use$meal[,3])),
                  heart=data.frame(day=as.numeric(date(dmy_hms(person4_use$heart[,1]))-date(dmy_hms(person4_use$glucose[1,1]))), 
                                   fiveminute=floor(daytime(dmy_hms(person4_use$heart[,1]))/5),
                                   heart=as.numeric(person4_use$heart[,2])),
                  gsr=data.frame(day=as.numeric(date(dmy_hms(person4_use$gsr[,1]))-date(dmy_hms(person4_use$glucose[1,1]))),
                                 fiveminute=floor(daytime(dmy_hms(person4_use$gsr[,1]))/5),
                                 gsr=as.numeric(person4_use$gsr[,2])),
                  skintemp=data.frame(day=as.numeric(date(dmy_hms(person4_use$skintemp[,1]))-date(dmy_hms(person4_use$glucose[1,1]))),
                                      fiveminute=floor(daytime(dmy_hms(person4_use$skintemp[,1]))/5),
                                      skintemp=as.numeric(person4_use$skintemp[,2])),
                  airtemp=data.frame(day=as.numeric(date(dmy_hms(person4_use$airtemp[,1]))-date(dmy_hms(person4_use$glucose[1,1]))),
                                     fiveminute=floor(daytime(dmy_hms(person4_use$airtemp[,1]))/5),
                                     airtemp=as.numeric(person4_use$airtemp[,2])),
                  steps=data.frame(day=as.numeric(date(dmy_hms(person4_use$steps[,1]))-date(dmy_hms(person4_use$glucose[1,1]))),
                                   fiveminute=floor(daytime(dmy_hms(person4_use$steps[,1]))/5),
                                   steps=as.numeric(person4_use$steps[,2]))
)

#deal with basal
nday=max(person4_5min$glucose$day)-min(person4_5min$glucose$day)+1
#deal with basal
date5min=data.frame(day=rep(seq(0,nday-1),each=288),fiveminute=rep(seq(0,287),nday))

tbasal=date5min
tbasal$rate=NA
ob=1
trate=0
person4_5min$basal=rbind(person4_5min$basal, c(max(person4_5min$glucose$day)+1,0,0))
for (i in 1:(288*nday)){
  if ((person4_5min$basal$day[ob]-min(person4_5min$glucose$day))*288+floor(person4_5min$basal$daytime[ob]/5)>i-1) {
    tbasal$rate[i]=trate
  } else{
    begin5minute=0
    tbasal$rate[i]=0
    while ((person4_5min$basal$day[ob]-min(person4_5min$glucose$day))*288+floor(person4_5min$basal$daytime[ob]/5)<=i-1) {
      tbasal$rate[i]=tbasal$rate[i]+ trate* (person4_5min$basal$daytime[ob]%%5-begin5minute)/5
      begin5minute=person4_5min$basal$daytime[ob]%%5
      trate=person4_5min$basal$rate[ob]
      ob=ob+1
    }
    tbasal$rate[i]=tbasal$rate[i]+ trate* (5-begin5minute)/5
  }
}
#adjust temp basal
for (i in 1:nrow(person4_5min$tempbasal)){
  begin5min_whole=(person4_5min$tempbasal$start_day[i]-min(person4_5min$glucose$day))*288+floor(person4_5min$tempbasal$start_daytime[i]/5)+1
  end5min_whole=(person4_5min$tempbasal$end_day[i]-min(person4_5min$glucose$day))*288+floor(person4_5min$tempbasal$end_daytime[i]/5)+1
  begin5min_rest=person4_5min$tempbasal$start_daytime[i]%%5
  end5min_rest=person4_5min$tempbasal$end_daytime[i]%%5
  trate=person4_5min$tempbasal$rate[i]
  while (begin5min_whole<end5min_whole){
    tbasal[begin5min_whole,]$rate=tbasal[begin5min_whole,]$rate*(begin5min_rest)/5+trate*(5-begin5min_rest)/5
    begin5min_whole=begin5min_whole+1
    begin5min_rest=0
  }
  tbasal[ begin5min_whole,]$rate=tbasal[ begin5min_whole,]$rate*(begin5min_rest+5-end5min_rest)/5 +trate*(end5min_rest-begin5min_rest)/5
}


#deal with bolus
tbolus=aggregate(dose~day+fiveminute, data=person4_5min$bolus,FUN=sum,na.rm=TRUE)
tbolus=merge(date5min,tbolus,by=c("day","fiveminute"),all.x=TRUE)
tbolus$lastmeal_day=NA
tbolus$lastmeal_daytime=NA
tbolus$lastmeal_carb=NA
tbolus$lastmeal_timediff=NA
tbolus$nextmeal_day=NA
tbolus$nextmeal_daytime=NA
tbolus$nextmeal_carb=NA
tbolus$nextmeal_timediff=NA

#add meal to the bolus dataset
lastmeal_day=NA
lastmeal_daytime=NA
lastmeal_carb=NA
nextmeal_day=person4_5min$meal$day[1]
nextmeal_daytime=person4_5min$meal$daytime[1]
nextmeal_carb=person4_5min$meal$carb[1]

person4_5min$meal=rbind(person4_5min$meal,c(max(person4_5min$glucose$day)+1,0,NA))

ob=1
for (day in min(person4_5min$glucose$day):max(person4_5min$glucose$day)){
  for (fiveminute in 0:287){
    i=day*288+fiveminute
    if ((day*1440+fiveminute*5)> (person4_5min$meal$day[ob]*1440+person4_5min$meal$daytime[ob])){
      lastmeal_day=person4_5min$meal$day[ob]
      lastmeal_daytime=person4_5min$meal$daytime[ob]
      lastmeal_carb=person4_5min$meal$carb[ob]
      ob=ob+1
    } 
    tbolus$lastmeal_day[i+1]=lastmeal_day
    tbolus$lastmeal_daytime[i+1]=lastmeal_daytime
    tbolus$lastmeal_carb[i+1]=lastmeal_carb
    tbolus$lastmeal_timediff[i+1]=(day*1440+fiveminute*5)-(lastmeal_day*1440+lastmeal_daytime)
    tbolus$nextmeal_day[i+1]=person4_5min$meal$day[ob]
    tbolus$nextmeal_daytime[i+1]=person4_5min$meal$daytime[ob]
    tbolus$nextmeal_carb[i+1]=person4_5min$meal$carb[ob]
    tbolus$nextmeal_timediff[i+1]=(person4_5min$meal$day[ob]*1440+person4_5min$meal$daytime[ob])-(day*1440+fiveminute*5) 
  }
}

#glucose
tglucose=aggregate(glucose~day+fiveminute,data=person4_5min$glucose,FUN=mean)
names(tglucose)=c("day","fiveminute","glucose")
tglucose=merge(date5min,tglucose,by=c("day","fiveminute"),all.x=TRUE)
tglucose$stability=ICG(tglucose$glucose)
tglucose$hyper=HyperIndex(tglucose$glucose)
tglucose$hypo=HypoIndex(tglucose$glucose)

theart=aggregate(heart~day+fiveminute, data=person4_5min$heart,FUN=mean)
tgsr=aggregate(gsr~day+fiveminute,data=person4_5min$gsr,FUN=mean)
tskintemp=aggregate(skintemp~day+fiveminute,data=person4_5min$skintemp,FUN=mean)
tairtemp=aggregate(airtemp~day+fiveminute,data=person4_5min$airtemp,FUN=mean)
tsteps=aggregate(steps~day+fiveminute,data=person4_5min$steps,FUN=mean)

person4_final=merge(tbasal,tbolus,by=c("day","fiveminute"),all=TRUE)
person4_final=merge(person4_final,tglucose,by=c("day","fiveminute"),all=TRUE)
person4_final=merge(person4_final,theart,by=c("day","fiveminute"),all=TRUE)
person4_final=merge(person4_final,tgsr,by=c("day","fiveminute"),all=TRUE)
person4_final=merge(person4_final,tskintemp,by=c("day","fiveminute"),all=TRUE)
person4_final=merge(person4_final,tairtemp,by=c("day","fiveminute"),all=TRUE)
person4_final=merge(person4_final,tsteps,by=c("day","fiveminute"),all=TRUE)



person4_final=person4_final[order(person4_final$day,person4_final$fiveminute),]
person4_final$dose[is.na(person4_final$dose)]=0

#delete observations that are missing too much

write.csv(person4_final, "fiveminute/person4-5min-train.csv",row.names=FALSE)


########################
########################
#######################
#person5################
#######################
person5_5min=list(glucose=data.frame( day=as.numeric(date(dmy_hms(person5_use$glucose[,1]))-date(dmy_hms(person5_use$glucose[1,1]))), 
                                      fiveminute=floor(daytime(dmy_hms(person5_use$glucose[,1]))/5),
                                      glucose=as.numeric(person5_use$glucose[,2])) ,
                  basal=data.frame(day=as.numeric(date(dmy_hms(person5_use$basal[,1]))-date(dmy_hms(person5_use$glucose[1,1]))), 
                                   daytime=daytime(dmy_hms(person5_use$basal[,1])),
                                   rate=as.numeric(person5_use$basal[,2])),
                  tempbasal=data.frame(start_day=as.numeric(date(dmy_hms(person5_use$tempbasal[,1]))-date(dmy_hms(person5_use$glucose[1,1]))),
                                       start_daytime=daytime(dmy_hms(person5_use$tempbasal[,1])),
                                       end_day=as.numeric(date(dmy_hms(person5_use$tempbasal[,2]))-date(dmy_hms(person5_use$glucose[1,1]))),
                                       end_daytime=daytime(dmy_hms(person5_use$tempbasal[,2])),
                                       rate=as.numeric(person5_use$tempbasal[,3])  ),
                  bolus=data.frame(day=as.numeric(date(dmy_hms(person5_use$bolus[,1]))-date(dmy_hms(person5_use$glucose[1,1]))), 
                                   fiveminute=floor(daytime(dmy_hms(person5_use$bolus[,1]))/5),
                                   dose=as.numeric(person5_use$bolus[,2])), 
                  meal=data.frame(day=as.numeric(date(dmy_hms(person5_use$meal[,1]))-date(dmy_hms(person5_use$glucose[1,1]))), 
                                  daytime=daytime(dmy_hms(person5_use$meal[,1])),
                                  carb=as.numeric(person5_use$meal[,3])),
                  heart=data.frame(day=as.numeric(date(dmy_hms(person5_use$heart[,1]))-date(dmy_hms(person5_use$glucose[1,1]))), 
                                   fiveminute=floor(daytime(dmy_hms(person5_use$heart[,1]))/5),
                                   heart=as.numeric(person5_use$heart[,2])),
                  gsr=data.frame(day=as.numeric(date(dmy_hms(person5_use$gsr[,1]))-date(dmy_hms(person5_use$glucose[1,1]))),
                                 fiveminute=floor(daytime(dmy_hms(person5_use$gsr[,1]))/5),
                                 gsr=as.numeric(person5_use$gsr[,2])),
                  skintemp=data.frame(day=as.numeric(date(dmy_hms(person5_use$skintemp[,1]))-date(dmy_hms(person5_use$glucose[1,1]))),
                                      fiveminute=floor(daytime(dmy_hms(person5_use$skintemp[,1]))/5),
                                      skintemp=as.numeric(person5_use$skintemp[,2])),
                  airtemp=data.frame(day=as.numeric(date(dmy_hms(person5_use$airtemp[,1]))-date(dmy_hms(person5_use$glucose[1,1]))),
                                     fiveminute=floor(daytime(dmy_hms(person5_use$airtemp[,1]))/5),
                                     airtemp=as.numeric(person5_use$airtemp[,2])),
                  steps=data.frame(day=as.numeric(date(dmy_hms(person5_use$steps[,1]))-date(dmy_hms(person5_use$glucose[1,1]))),
                                   fiveminute=floor(daytime(dmy_hms(person5_use$steps[,1]))/5),
                                   steps=as.numeric(person5_use$steps[,2]))
)

#deal with basal
nday=max(person5_5min$glucose$day)-min(person5_5min$glucose$day)+1
#deal with basal
date5min=data.frame(day=rep(seq(0,nday-1),each=288),fiveminute=rep(seq(0,287),nday))

tbasal=date5min
tbasal$rate=NA
ob=1
trate=0
person5_5min$basal=rbind(person5_5min$basal, c(max(person5_5min$glucose$day)+1,0,0))
for (i in 1:(288*nday)){
  if ((person5_5min$basal$day[ob]-min(person5_5min$glucose$day))*288+floor(person5_5min$basal$daytime[ob]/5)>i-1) {
    tbasal$rate[i]=trate
  } else{
    begin5minute=0
    tbasal$rate[i]=0
    while ((person5_5min$basal$day[ob]-min(person5_5min$glucose$day))*288+floor(person5_5min$basal$daytime[ob]/5)<=i-1) {
      tbasal$rate[i]=tbasal$rate[i]+ trate* (person5_5min$basal$daytime[ob]%%5-begin5minute)/5
      begin5minute=person5_5min$basal$daytime[ob]%%5
      trate=person5_5min$basal$rate[ob]
      ob=ob+1
    }
    tbasal$rate[i]=tbasal$rate[i]+ trate* (5-begin5minute)/5
  }
}
#adjust temp basal
for (i in 1:nrow(person5_5min$tempbasal)){
  begin5min_whole=(person5_5min$tempbasal$start_day[i]-min(person5_5min$glucose$day))*288+floor(person5_5min$tempbasal$start_daytime[i]/5)+1
  end5min_whole=(person5_5min$tempbasal$end_day[i]-min(person5_5min$glucose$day))*288+floor(person5_5min$tempbasal$end_daytime[i]/5)+1
  begin5min_rest=person5_5min$tempbasal$start_daytime[i]%%5
  end5min_rest=person5_5min$tempbasal$end_daytime[i]%%5
  trate=person5_5min$tempbasal$rate[i]
  while (begin5min_whole<end5min_whole){
    tbasal[begin5min_whole,]$rate=tbasal[begin5min_whole,]$rate*(begin5min_rest)/5+trate*(5-begin5min_rest)/5
    begin5min_whole=begin5min_whole+1
    begin5min_rest=0
  }
  tbasal[ begin5min_whole,]$rate=tbasal[ begin5min_whole,]$rate*(begin5min_rest+5-end5min_rest)/5 +trate*(end5min_rest-begin5min_rest)/5
}


#deal with bolus
tbolus=aggregate(dose~day+fiveminute, data=person5_5min$bolus,FUN=sum,na.rm=TRUE)
tbolus=merge(date5min,tbolus,by=c("day","fiveminute"),all.x=TRUE)
tbolus$lastmeal_day=NA
tbolus$lastmeal_daytime=NA
tbolus$lastmeal_carb=NA
tbolus$lastmeal_timediff=NA
tbolus$nextmeal_day=NA
tbolus$nextmeal_daytime=NA
tbolus$nextmeal_carb=NA
tbolus$nextmeal_timediff=NA

#add meal to the bolus dataset
lastmeal_day=NA
lastmeal_daytime=NA
lastmeal_carb=NA
nextmeal_day=person5_5min$meal$day[1]
nextmeal_daytime=person5_5min$meal$daytime[1]
nextmeal_carb=person5_5min$meal$carb[1]

person5_5min$meal=rbind(person5_5min$meal,c(max(person5_5min$glucose$day)+1,0,NA))

ob=1
for (day in min(person5_5min$glucose$day):max(person5_5min$glucose$day)){
  for (fiveminute in 0:287){
    i=day*288+fiveminute
    if ((day*1440+fiveminute*5)> (person5_5min$meal$day[ob]*1440+person5_5min$meal$daytime[ob])){
      lastmeal_day=person5_5min$meal$day[ob]
      lastmeal_daytime=person5_5min$meal$daytime[ob]
      lastmeal_carb=person5_5min$meal$carb[ob]
      ob=ob+1
    } 
    tbolus$lastmeal_day[i+1]=lastmeal_day
    tbolus$lastmeal_daytime[i+1]=lastmeal_daytime
    tbolus$lastmeal_carb[i+1]=lastmeal_carb
    tbolus$lastmeal_timediff[i+1]=(day*1440+fiveminute*5)-(lastmeal_day*1440+lastmeal_daytime)
    tbolus$nextmeal_day[i+1]=person5_5min$meal$day[ob]
    tbolus$nextmeal_daytime[i+1]=person5_5min$meal$daytime[ob]
    tbolus$nextmeal_carb[i+1]=person5_5min$meal$carb[ob]
    tbolus$nextmeal_timediff[i+1]=(person5_5min$meal$day[ob]*1440+person5_5min$meal$daytime[ob])-(day*1440+fiveminute*5) 
  }
}

#glucose
tglucose=aggregate(glucose~day+fiveminute,data=person5_5min$glucose,FUN=mean)
names(tglucose)=c("day","fiveminute","glucose")
tglucose=merge(date5min,tglucose,by=c("day","fiveminute"),all.x=TRUE)
tglucose$stability=ICG(tglucose$glucose)
tglucose$hyper=HyperIndex(tglucose$glucose)
tglucose$hypo=HypoIndex(tglucose$glucose)

theart=aggregate(heart~day+fiveminute, data=person5_5min$heart,FUN=mean)
tgsr=aggregate(gsr~day+fiveminute,data=person5_5min$gsr,FUN=mean)
tskintemp=aggregate(skintemp~day+fiveminute,data=person5_5min$skintemp,FUN=mean)
tairtemp=aggregate(airtemp~day+fiveminute,data=person5_5min$airtemp,FUN=mean)
tsteps=aggregate(steps~day+fiveminute,data=person5_5min$steps,FUN=mean)

person5_final=merge(tbasal,tbolus,by=c("day","fiveminute"),all=TRUE)
person5_final=merge(person5_final,tglucose,by=c("day","fiveminute"),all=TRUE)
person5_final=merge(person5_final,theart,by=c("day","fiveminute"),all=TRUE)
person5_final=merge(person5_final,tgsr,by=c("day","fiveminute"),all=TRUE)
person5_final=merge(person5_final,tskintemp,by=c("day","fiveminute"),all=TRUE)
person5_final=merge(person5_final,tairtemp,by=c("day","fiveminute"),all=TRUE)
person5_final=merge(person5_final,tsteps,by=c("day","fiveminute"),all=TRUE)



person5_final=person5_final[order(person5_final$day,person5_final$fiveminute),]
person5_final$dose[is.na(person5_final$dose)]=0

#delete observations that are missing too much

write.csv(person5_final, "fiveminute/person5-5min-train.csv",row.names=FALSE)


########################
########################
#######################
#person6################
#######################
person6_5min=list(glucose=data.frame( day=as.numeric(date(dmy_hms(person6_use$glucose[,1]))-date(dmy_hms(person6_use$glucose[1,1]))), 
                                      fiveminute=floor(daytime(dmy_hms(person6_use$glucose[,1]))/5),
                                      glucose=as.numeric(person6_use$glucose[,2])) ,
                  basal=data.frame(day=as.numeric(date(dmy_hms(person6_use$basal[,1]))-date(dmy_hms(person6_use$glucose[1,1]))), 
                                   daytime=daytime(dmy_hms(person6_use$basal[,1])),
                                   rate=as.numeric(person6_use$basal[,2])),
                  tempbasal=data.frame(start_day=as.numeric(date(dmy_hms(person6_use$tempbasal[,1]))-date(dmy_hms(person6_use$glucose[1,1]))),
                                       start_daytime=daytime(dmy_hms(person6_use$tempbasal[,1])),
                                       end_day=as.numeric(date(dmy_hms(person6_use$tempbasal[,2]))-date(dmy_hms(person6_use$glucose[1,1]))),
                                       end_daytime=daytime(dmy_hms(person6_use$tempbasal[,2])),
                                       rate=as.numeric(person6_use$tempbasal[,3])  ),
                  bolus=data.frame(day=as.numeric(date(dmy_hms(person6_use$bolus[,1]))-date(dmy_hms(person6_use$glucose[1,1]))), 
                                   fiveminute=floor(daytime(dmy_hms(person6_use$bolus[,1]))/5),
                                   dose=as.numeric(person6_use$bolus[,2])), 
                  meal=data.frame(day=as.numeric(date(dmy_hms(person6_use$meal[,1]))-date(dmy_hms(person6_use$glucose[1,1]))), 
                                  daytime=daytime(dmy_hms(person6_use$meal[,1])),
                                  carb=as.numeric(person6_use$meal[,3])),
                  heart=data.frame(day=as.numeric(date(dmy_hms(person6_use$heart[,1]))-date(dmy_hms(person6_use$glucose[1,1]))), 
                                   fiveminute=floor(daytime(dmy_hms(person6_use$heart[,1]))/5),
                                   heart=as.numeric(person6_use$heart[,2])),
                  gsr=data.frame(day=as.numeric(date(dmy_hms(person6_use$gsr[,1]))-date(dmy_hms(person6_use$glucose[1,1]))),
                                 fiveminute=floor(daytime(dmy_hms(person6_use$gsr[,1]))/5),
                                 gsr=as.numeric(person6_use$gsr[,2])),
                  skintemp=data.frame(day=as.numeric(date(dmy_hms(person6_use$skintemp[,1]))-date(dmy_hms(person6_use$glucose[1,1]))),
                                      fiveminute=floor(daytime(dmy_hms(person6_use$skintemp[,1]))/5),
                                      skintemp=as.numeric(person6_use$skintemp[,2])),
                  airtemp=data.frame(day=as.numeric(date(dmy_hms(person6_use$airtemp[,1]))-date(dmy_hms(person6_use$glucose[1,1]))),
                                     fiveminute=floor(daytime(dmy_hms(person6_use$airtemp[,1]))/5),
                                     airtemp=as.numeric(person6_use$airtemp[,2])),
                  steps=data.frame(day=as.numeric(date(dmy_hms(person6_use$steps[,1]))-date(dmy_hms(person6_use$glucose[1,1]))),
                                   fiveminute=floor(daytime(dmy_hms(person6_use$steps[,1]))/5),
                                   steps=as.numeric(person6_use$steps[,2]))
)

#deal with basal
nday=max(person6_5min$glucose$day)-min(person6_5min$glucose$day)+1
#deal with basal
date5min=data.frame(day=rep(seq(0,nday-1),each=288),fiveminute=rep(seq(0,287),nday))

tbasal=date5min
tbasal$rate=NA
ob=1
trate=0
person6_5min$basal=rbind(person6_5min$basal, c(max(person6_5min$glucose$day)+1,0,0))
for (i in 1:(288*nday)){
  if ((person6_5min$basal$day[ob]-min(person6_5min$glucose$day))*288+floor(person6_5min$basal$daytime[ob]/5)>i-1) {
    tbasal$rate[i]=trate
  } else{
    begin5minute=0
    tbasal$rate[i]=0
    while ((person6_5min$basal$day[ob]-min(person6_5min$glucose$day))*288+floor(person6_5min$basal$daytime[ob]/5)<=i-1) {
      tbasal$rate[i]=tbasal$rate[i]+ trate* (person6_5min$basal$daytime[ob]%%5-begin5minute)/5
      begin5minute=person6_5min$basal$daytime[ob]%%5
      trate=person6_5min$basal$rate[ob]
      ob=ob+1
    }
    tbasal$rate[i]=tbasal$rate[i]+ trate* (5-begin5minute)/5
  }
}
#adjust temp basal
for (i in 1:nrow(person6_5min$tempbasal)){
  begin5min_whole=(person6_5min$tempbasal$start_day[i]-min(person6_5min$glucose$day))*288+floor(person6_5min$tempbasal$start_daytime[i]/5)+1
  end5min_whole=(person6_5min$tempbasal$end_day[i]-min(person6_5min$glucose$day))*288+floor(person6_5min$tempbasal$end_daytime[i]/5)+1
  begin5min_rest=person6_5min$tempbasal$start_daytime[i]%%5
  end5min_rest=person6_5min$tempbasal$end_daytime[i]%%5
  trate=person6_5min$tempbasal$rate[i]
  while (begin5min_whole<end5min_whole){
    tbasal[begin5min_whole,]$rate=tbasal[begin5min_whole,]$rate*(begin5min_rest)/5+trate*(5-begin5min_rest)/5
    begin5min_whole=begin5min_whole+1
    begin5min_rest=0
  }
  tbasal[ begin5min_whole,]$rate=tbasal[ begin5min_whole,]$rate*(begin5min_rest+5-end5min_rest)/5 +trate*(end5min_rest-begin5min_rest)/5
}


#deal with bolus
tbolus=aggregate(dose~day+fiveminute, data=person6_5min$bolus,FUN=sum,na.rm=TRUE)
tbolus=merge(date5min,tbolus,by=c("day","fiveminute"),all.x=TRUE)
tbolus$lastmeal_day=NA
tbolus$lastmeal_daytime=NA
tbolus$lastmeal_carb=NA
tbolus$lastmeal_timediff=NA
tbolus$nextmeal_day=NA
tbolus$nextmeal_daytime=NA
tbolus$nextmeal_carb=NA
tbolus$nextmeal_timediff=NA

#add meal to the bolus dataset
lastmeal_day=NA
lastmeal_daytime=NA
lastmeal_carb=NA
nextmeal_day=person6_5min$meal$day[1]
nextmeal_daytime=person6_5min$meal$daytime[1]
nextmeal_carb=person6_5min$meal$carb[1]

person6_5min$meal=rbind(person6_5min$meal,c(max(person6_5min$glucose$day)+1,0,NA))

ob=1
for (day in min(person6_5min$glucose$day):max(person6_5min$glucose$day)){
  for (fiveminute in 0:287){
    i=day*288+fiveminute
    if ((day*1440+fiveminute*5)> (person6_5min$meal$day[ob]*1440+person6_5min$meal$daytime[ob])){
      lastmeal_day=person6_5min$meal$day[ob]
      lastmeal_daytime=person6_5min$meal$daytime[ob]
      lastmeal_carb=person6_5min$meal$carb[ob]
      ob=ob+1
    } 
    tbolus$lastmeal_day[i+1]=lastmeal_day
    tbolus$lastmeal_daytime[i+1]=lastmeal_daytime
    tbolus$lastmeal_carb[i+1]=lastmeal_carb
    tbolus$lastmeal_timediff[i+1]=(day*1440+fiveminute*5)-(lastmeal_day*1440+lastmeal_daytime)
    tbolus$nextmeal_day[i+1]=person6_5min$meal$day[ob]
    tbolus$nextmeal_daytime[i+1]=person6_5min$meal$daytime[ob]
    tbolus$nextmeal_carb[i+1]=person6_5min$meal$carb[ob]
    tbolus$nextmeal_timediff[i+1]=(person6_5min$meal$day[ob]*1440+person6_5min$meal$daytime[ob])-(day*1440+fiveminute*5) 
  }
}

#glucose
tglucose=aggregate(glucose~day+fiveminute,data=person6_5min$glucose,FUN=mean)
names(tglucose)=c("day","fiveminute","glucose")
tglucose=merge(date5min,tglucose,by=c("day","fiveminute"),all.x=TRUE)
tglucose$stability=ICG(tglucose$glucose)
tglucose$hyper=HyperIndex(tglucose$glucose)
tglucose$hypo=HypoIndex(tglucose$glucose)

theart=aggregate(heart~day+fiveminute, data=person6_5min$heart,FUN=mean)
tgsr=aggregate(gsr~day+fiveminute,data=person6_5min$gsr,FUN=mean)
tskintemp=aggregate(skintemp~day+fiveminute,data=person6_5min$skintemp,FUN=mean)
tairtemp=aggregate(airtemp~day+fiveminute,data=person6_5min$airtemp,FUN=mean)
tsteps=aggregate(steps~day+fiveminute,data=person6_5min$steps,FUN=mean)

person6_final=merge(tbasal,tbolus,by=c("day","fiveminute"),all=TRUE)
person6_final=merge(person6_final,tglucose,by=c("day","fiveminute"),all=TRUE)
person6_final=merge(person6_final,theart,by=c("day","fiveminute"),all=TRUE)
person6_final=merge(person6_final,tgsr,by=c("day","fiveminute"),all=TRUE)
person6_final=merge(person6_final,tskintemp,by=c("day","fiveminute"),all=TRUE)
person6_final=merge(person6_final,tairtemp,by=c("day","fiveminute"),all=TRUE)
person6_final=merge(person6_final,tsteps,by=c("day","fiveminute"),all=TRUE)



person6_final=person6_final[order(person6_final$day,person6_final$fiveminute),]
person6_final$dose[is.na(person6_final$dose)]=0

#delete observations that are missing too much

write.csv(person6_final, "fiveminute/person6-5min-train.csv",row.names=FALSE)

remove(data_var)
remove(person1)
remove(person1_use)
remove(person2)
remove(person2_use)
remove(person3)
remove(person3_use)
remove(person4)
remove(person4_use)
remove(person5)
remove(person5_use)
remove(person6)
remove(person6_use)
remove(tbasal)
remove(tbolus)
remove(tglucose)
remove(theart)
remove(tgsr)
remove(tskintemp)
remove(tairtemp)
remove(tsteps)

remove(data)
remove(begin5min_whole)
remove(begin5min_rest)
remove(end5min_whole)
remove(end5min_rest)
remove(day)
remove(fiveminute)
remove(i)
remove(lastmeal_day)
remove(lastmeal_daytime)
remove(lastmeal_carb)
remove(nextmeal_day)
remove(nextmeal_daytime)
remove(nextmeal_carb)
remove(nday)
remove(ob)
remove(rootnode)
remove(trate)
remove(vari)


person1_temp=person1_final
person1_temp$day=person1_temp$day+10000
person2_temp=person2_final
person2_temp$day=person2_temp$day+20000
person3_temp=person3_final
person3_temp$day=person3_temp$day+30000
person4_temp=person4_final
person4_temp$day=person4_temp$day+40000
person5_temp=person5_final
person5_temp$day=person5_temp$day+50000
person6_temp=person6_final
person6_temp$day=person6_temp$day+60000
combined=rbind(person1_temp,person2_temp,person3_temp,person4_temp,person5_temp,person6_temp)

remove(person1_temp)
remove(person2_temp)
remove(person3_temp)
remove(person4_temp)
remove(person5_temp)
remove(person6_temp)

person1_train=person1_final
person2_train=person2_final
person3_train=person3_final
person4_train=person4_final
person5_train=person5_final
person6_train=person6_final
combined_train=combined
remove(person1_final)
remove(person2_final)
remove(person3_final)
remove(person4_final)
remove(person5_final)
remove(person6_final)
remove(combined)
save.image("fiveminute/data_transformed-5min-train.Rdata")

