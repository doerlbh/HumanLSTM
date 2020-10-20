rm(list=ls())
# -----------------------------------------------------------------------------
# 0. Create data --------------------------------------------------------------
# -----------------------------------------------------------------------------
source("dataPreProc/0-am1993.R")
# each of these files reads in the raw data from dataPreProc/DATA/ subfolder
# and then saves a data.frame inside that folder with the data. then next
# we load all that in. doing it this way makes sure there are no side-effects
# where objects created from one data preprocessing might affect another
# because each of these files begins and ends with rm(list=ls())

# -----------------------------------------------------------------------------
# I. Merge data ---------------------------------------------------------------
# -----------------------------------------------------------------------------

load("dataPreProc/DATA/AM.Rda")
pdGames <- AM

# -----------------------------------------------------------------------------
# III. Create "history-of-play" features --------------------------------------
# -----------------------------------------------------------------------------

# creating a lag
lagCreate <- function(vec, lag=1) { 
        # vec is any numeric vector
        # default is lag 1
        nulls <- rep(NA, lag)
        embed(c(nulls, vec), (lag+1))[,(lag+1)]
}

for (i in 1:9) { #create 9 lags of my.decision
        columnName <- paste("my.decision", i, sep = "")
        pdGames[[columnName]] <- lagCreate(pdGames$my.decision, lag=i)
        for (j in i:1) {
                pdGames[pdGames$period==j, ][[columnName]] <- NA
        } # this loop puts the NA's into the periods that need them
} # so for e.g. "my.decision1" means the 1-period-lag of my.decision

for (i in 1:9) { #create 9 lags of other.decision
        columnName <- paste("other.decision", i, sep = "")
        pdGames[[columnName]] <- lagCreate(pdGames$other.decision, lag=i)
        for (j in i:1) {
                pdGames[pdGames$period==j, ][[columnName]] <- NA
        } # this loop puts the NA's into the periods that need them
} 

for (i in 1:9) { #create 9 lags of my.payoff
        columnName <- paste("my.payoff", i, sep = "")
        pdGames[[columnName]] <- lagCreate(pdGames$my.payoff, lag=i)
        for (j in i:1) {
                pdGames[pdGames$period==j, ][[columnName]] <- NA
        } # this loop puts the NA's into the periods that need them
} 

for (i in 1:9) { #create 9 lags of other.payoff
        columnName <- paste("other.payoff", i, sep = "")
        pdGames[[columnName]] <- lagCreate(pdGames$other.payoff, lag=i)
        for (j in i:1) {
                pdGames[pdGames$period==j, ][[columnName]] <- NA
        } # this loop puts the NA's into the periods that need them
}

# get rid of the columns that are giving data about the current period t
# bc these cannot be used to predict what will happen in t: my.decision(t)
pd <- pdGames[, -which(names(pdGames) %in% 
                         c("other.decision", "my.payoff", "other.payoff"))]

# make the outcome variable a factor and assign the 0 to defect and the 1 to coop
pd$my.decision <- factor(pd$my.decision, labels = c("defect", "coop"))
levels(pd$my.decision)

final <- pd

save(final, file="dataPreProc/DATA/final.Rda")
rm(list=ls())
