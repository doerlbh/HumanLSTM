
load("dataPreProc/DATA/final.Rda")
STAT <- "mean"
# doing it this way makes sure there are no side-effects
# where objects created from one data preprocessing might affect another
# because each of these files begins and ends with rm(list=ls())

periods_to_compare_for_dynamics <- 8
period_vec_create <- function(datasubset, 
                              periods=periods_to_compare_for_dynamics){
  period_vec <- rep(NA, length(periods))
  for (i in seq(periods)){
    if (nrow(datasubset[datasubset$period==i, ]) > 0){ 
      period_vec[i] <- do.call(STAT, list(x = 
                                            ifelse(datasubset[datasubset$period==i, 
                                                              which(names(datasubset) %in% "my.decision")]=="coop", 1, 0)))
    } else{
      period_vec[i] <- NA
    }
  }
  stopifnot(length(period_vec)==periods)
  period_vec
}

# ceatings all vecs here and then at end cbind()ing them to create a df.
n <- 1 # groups
data <- rep(NA, n)
error <- rep(NA, n)
delta <- rep(NA, n)
infin <- rep(NA, n)
contin <- rep(NA, n)
risk <- rep(NA, n)
r <- rep(NA, n)
s <- rep(NA, n)
t <- rep(NA, n)
p <- rep(NA, n)
r1 <- rep(NA, n)
r2 <- rep(NA, n)

coop <- rep(NA, n)
period_mat <- data.frame(matrix(NA, nrow=n, ncol=periods_to_compare_for_dynamics))
colnames(period_mat) <- seq(periods_to_compare_for_dynamics)

index <- as.list(rep(NA, n))
payoffs <- as.list(rep(NA, n))
names(index) <- seq(n)

##########################################
##### AM #####
i <- 1
ra <- 0.07
stopifnot(ra %in% unique(final[final$data=='AM', "my.payoff1"]))
sa <- 0.00
stopifnot(sa %in% unique(final[final$data=='AM', "my.payoff1"]))
ta <- 0.12
stopifnot(ta %in% unique(final[final$data=='AM', "my.payoff1"]))
pa <- 0.04
stopifnot(pa %in% unique(final[final$data=='AM', "my.payoff1"]))
payoffs[[i]] <- matrix(c(ra, sa,
                         ta, pa), 
                       2, 2, byrow=T)
index[[i]] <- final$data=='AM'
data[i] <- "AM"

##########################################
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

add_game_num <- function(d){
  game <- rep(NA, nrow(d))
  game[1] <- 1
  for (i in 2:nrow(d)){
    game[i] <- ifelse(d$period[i]==1, game[i-1] + 1, game[i-1])
  }
  gamemax <- rep(NA, nrow(d))
  for (i in 2:nrow(d)){
    if(d$period[i]==1)
      gamemax[(i-1):(i-d$period[i-1])] <- d$period[i-1]
  }
  gamemax[nrow(d):(nrow(d)-d$period[nrow(d)])] <- d$period[nrow(d)]
  gamemax
}

for (i in seq(n)){
  error[i] <- mean(final[index[[i]], which(names(final) %in% "error")]) 
  delta[i] <- mean(final[index[[i]], which(names(final) %in% "delta")]) # 0.9 if delta = 0 at period 10
  # if more than one gamemax number then it must be an infinite game to have randomness in the lengths:
  infin[i] <- ifelse(length(unique(add_game_num(final[index[[i]], ]))) > 1, 1, 0)
  
  contin[i] <- ifelse(i %in% 19:22, 1, 0) # the FO data is contin==1
  
  risk[i] <- Mode(final[index[[i]], which(names(final) %in% "risk")]) # Mode for risk bc:  
  # only one entry that is not 1 or 0 and keeping it as a factor rather than numeric is good for modeling
  
  stopifnot(length(unique(final[index[[i]], which(names(final) %in% "r1")]) ) == 1)
  r1[i] <- unique(final[index[[i]], which(names(final) %in% "r1")]) 
  stopifnot(length(unique(final[index[[i]], which(names(final) %in% "r2")]) ) == 1)
  r2[i] <- unique(final[index[[i]], which(names(final) %in% "r2")])
  
  r[i] <- payoffs[[i]][1,1] - payoffs[[i]][which(payoffs[[i]]==min(payoffs[[i]]))]
  s[i] <- payoffs[[i]][1,2] - payoffs[[i]][which(payoffs[[i]]==min(payoffs[[i]]))]
  t[i] <- payoffs[[i]][2,1] - payoffs[[i]][which(payoffs[[i]]==min(payoffs[[i]]))]
  p[i] <- payoffs[[i]][2,2] - payoffs[[i]][which(payoffs[[i]]==min(payoffs[[i]]))]
  coop[i] <- sum(final[index[[i]], which(names(final) %in% "my.decision")]=="coop")/nrow(final[index[[i]], ])
  period_mat[i, ] <- period_vec_create(final[index[[i]], ])
}
##########################################
patterns <- data.frame(error=error, delta=delta, infin = infin, 
                       contin = contin,
                       risk=risk, r1 = r1, r2 = r2,
                       r=r, s=s, t=t, p=p, coop=coop, 
                       data=as.factor(data),
                       group=seq(n))
patterns <- cbind(patterns, period_mat)

################################################################################
################################################################################
# NOW ADD THE RSTP TO THE final.Rda DATA
r <- rep(NA, nrow(final))
s <- rep(NA, nrow(final))
t <- rep(NA, nrow(final))
p <- rep(NA, nrow(final))

##########################################
##### AM #####
AMpayoffs <- matrix(c(ra, sa,
                         ta, pa), 
                       2, 2, byrow=T)
r[final$data=='AM'] <- AMpayoffs[1,1]- AMpayoffs[which(AMpayoffs==min(AMpayoffs))]
s[final$data=='AM'] <- AMpayoffs[1,2]- AMpayoffs[which(AMpayoffs==min(AMpayoffs))]
t[final$data=='AM'] <- AMpayoffs[2,1]- AMpayoffs[which(AMpayoffs==min(AMpayoffs))]
p[final$data=='AM'] <- AMpayoffs[2,2]- AMpayoffs[which(AMpayoffs==min(AMpayoffs))]


group <- rep(NA, nrow(final))
for (i in as.numeric(names(index))){
  group[index[[i]]] <- i
}

infin <- rep(NA, nrow(final))
for (i in seq(nrow(patterns))){
  infin[index[[i]]] <- patterns$infin[i]
}

contin <- rep(NA, nrow(final))
for (i in seq(nrow(patterns))){
  contin[index[[i]]] <- patterns$contin[i]
}

data <- cbind(final, r, s, t, p, infin, contin, group)
# data <- data[, -which(names(data) %in% c("r1", "r2"))]
save(data, file="data.Rda")

patterns <- cbind(patterns, decisions = as.numeric(table(data$group))) # how many decisions in the data for each group
save(patterns, file="patterns.Rda")

rm(list=ls())
