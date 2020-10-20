rm(list=ls())

d1 <- read.table("dataPreProc/DATA/Andreoni_Miller_1993/PARTNO.DAT", header=F, skip=5)
# readLines("dataPreProc/DATA/Andreoni_Miller_1993/PARTNO.DAT", n=10)[5]
names(d1) <- c("round", "subj", "partner", "my.decision", "other.decision", "my.payoff", "total.payoff")

# playing computer in d2, so we are not using this bc we are only looking at real human play against real human opponents.
# d2 <- read.table("dataPreProc/DATA/Andreoni_Miller_1993/PARTNO2.DAT", header=F, skip=5)
# dnames2 <- readLines("dataPreProc/DATA/Andreoni_Miller_1993/PARTNO2.DAT", n=10)[5]

game <- numeric()
for (i in 1:20) game <- c(game, rep(i, 140))
stopifnot(length(game)==nrow(d1))
d <- cbind(d1, game=game)

player <- data.frame()
# match is a list filled with max(d$game) data.frames that each have dim() == 14, 2, col1 is subj and col2 is partner
match <- as.list(rep(NA, max(d$game)))
for (g in unique(d$game)){
  # subj-partner matchings for that game
  match[[g]] <- unique(d[d$game==g, which(names(d) %in% c("subj", "partner"))])
  for (i in seq(nrow(match[[g]]))){
    player <<- rbind(player, d[d$game==g & as.numeric(match[[g]][i, 1]) == d[d$game==g, which(names(d) %in% c("subj"))] & 
                                as.numeric(match[[g]][i, 2]) == d[d$game==g, which(names(d) %in% c("partner"))], ])
  }
}

period <- rep(1:10, 280)
d <- cbind(period, player[ , which(names(d) %in% c("my.decision", "other.decision", "my.payoff", "total.payoff"))])

if(nrow(d)!=nrow(d1)) stop("Did not create am1993 data correctly.")

# In this data 1 is defect and 0 is cooperate, but we want the opposite, so swap:
d$my.decision <- ifelse(d$my.decision==1, 0, 1)
d$other.decision <- ifelse(d$other.decision==1, 0, 1)

period_vec_create <- function(data, periods){
  period_vec <- rep(NA, length(periods))
  for (i in seq(periods)){
    if (nrow(data[data$period==i, ]) > 0){ 
      period_vec[i] <- mean(data[data$period==i, which(names(data) %in% "my.decision")], na.rm=TRUE)
    } else{
      period_vec[i] <- NA
      warning(paste("You have a period vector where period", i, "does not have any data."))
    }
  }
  stopifnot(length(period_vec)==periods)
  period_vec
}
# to make sure we created the dataset correctly look at graph in am1993 paper pg. 576 the Economic Journal
# and compare it to this data:
# period_vec_create(d, 10)
d$other.payoff <- d$total.payoff - d$my.payoff

n <- nrow(d)
error <- rep(0, n)
delta <- rep(0.9, n)
risk <- rep(0, n)
r <- rep(0.07, n)
stopifnot(unique(r) %in% unique(d$my.payoff))
s <- rep(0.00, n)
stopifnot(unique(s) %in% unique(d$my.payoff))
t <- rep(0.12, n)
stopifnot(unique(t) %in% unique(d$my.payoff))
p <- rep(0.04, n)
stopifnot(unique(p) %in% unique(d$my.payoff))
data <- rep("AM", n)

r1 <- rep(0.25, n)
r2 <- rep(0.583, n)
  
AM <- data.frame(period = d$period, 
                 my.decision = d$my.decision, other.decision = d$other.decision,
                 my.payoff = d$my.payoff,  other.payoff = d$other.payoff,
                 error = error, delta = delta, risk = risk,
                 r1 = r1, r2 = r2,
                 data = data)

stopifnot(!anyNA(AM))
save(AM, file="dataPreProc/DATA/AM.Rda") 
rm(list=ls())
