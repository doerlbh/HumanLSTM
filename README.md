# HumanLSTM



Code for our paper: 

**"Predicting Human Decision Making in Psychological Tasks with Recurrent Neural Networks"** 

by [Baihan Lin](https://www.baihan.nyc/) (Columbia University), [Djallel Bouneffouf](https://scholar.google.com/citations?user=i2a1LUMAAAAJ&hl=en) (IBM Research), [Guillermo Cecchi](https://researcher.watson.ibm.com/researcher/view.php?person=us-gcecchi) (IBM Research).





For the latest full paper: https://doi.org/10.1371/journal.pone.0267907



All the experimental results can be reproduced using the code in this repository. Feel free to contact me by doerlbh@gmail.com if you have any question about our work.



**Abstract**

Unlike traditional time series, the action sequences of human decision making usually involve many cognitive processes such as beliefs, desires, intentions, and theory of mind, i.e., what others are thinking. This makes predicting human decision-making challenging to be treated agnostically to the underlying psychological mechanisms. We propose here to use a recurrent neural network architecture based on long short-term memory networks (LSTM) to predict the time series of the actions taken by human subjects engaged in gaming activity, the first application of such methods in this research domain. In this study, we collate the human data from 8 published literature of the Iterated Prisoner’s Dilemma comprising 168,386 individual decisions and post-process them into 8,257 behavioral trajectories of 9 actions each for both players. Similarly, we collate 617 trajectories of 95 actions from 10 different published studies of Iowa Gambling Task experiments with healthy human subjects. We train our prediction networks on the behavioral data and demonstrate a clear advantage over the state-of-the-art methods in predicting human decision-making trajectories in both the single-agent scenario of the Iowa Gambling Task and the multi-agent scenario of the Iterated Prisoner’s Dilemma. Moreover, we observe that the weights of the LSTM networks modeling the top performers tend to have a wider distribution compared to poor performers, as well as a larger bias, which suggest possible interpretations for the distribution of strategies adopted by each group.


## Info

Language: Python3


Platform: MacOS, Linux, Windows

by Baihan Lin, August 2020

  

  


## Citation

If you find this work helpful, please try the models out and cite our works. Thanks!

    @article{10.1371/journal.pone.0267907,
    doi = {10.1371/journal.pone.0267907},
    author = {Lin, Baihan AND Bouneffouf, Djallel AND Cecchi, Guillermo},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {Predicting human decision making in psychological tasks with recurrent neural networks},
    year = {2022},
    month = {05},
    volume = {17},
    url = {https://doi.org/10.1371/journal.pone.0267907},
    pages = {1-18},
    number = {5},
    }


## Tasks

* Predict human behavioral trajectories in Iterated Gambling Task (IGT)

* Predict human behavioral trajectories in Iterated Prisoner's Dilemma (IPD)

  

## Algorithms:

* LSTM

* autoregression

* logistic regression

  

## Requirements

* numpy, scikit-learn, scipy, pandas, PyTorch, statsmodels



## Related work and repositories

* "Online Learning in Iterated Prisoner's Dilemma to Mimic Human Behavior" at https://github.com/doerlbh/dilemmaRL
* "A Story of Two Streams: Reinforcement Learning Models from Human Behavior and Neuropsychiatry" at https://github.com/doerlbh/mentalRL



