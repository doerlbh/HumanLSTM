To create the data used for the Nay and Vorobeychik paper "Predicting Human Cooperation" from the original data from 

Andreoni J, Miller JH. Rational cooperation in the finitely repeated prisoner’s dilemma: Experimental evidence. The Economic Journal. 1993;103: 570–585. doi:10.2307/2234532

run the file main.R, e.g. from the command line type "Rscript main.R". This runs the file 1-merge.R and then 1-patterns.R which creates the individual level data in a data.frame saved to disk called "data.Rda" and the the summary statistics of play in a data.frame called "patterns.Rda". This data is used to estimate the models in Nay and Vorobeychik.

Andreoni and Miller shared their underlying data from their experiments and we downloaded that.

The original data from the Andreoni and Miller 1993 paper, which is in the dataPreProc/DATA/Andreoni_Miller_1993 folder was downloaded from this website: http://econlab.ucsd.edu/getdata/.