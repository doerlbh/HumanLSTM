# HumanLSTM



Code for our paper: 

**"Predicting Human Decision Making in Psychological Tasks with Recurrent Neural Networks"** 

by [Baihan Lin](https://www.baihan.nyc/) (Columbia), [Djallel Bouneffouf](https://scholar.google.com/citations?user=i2a1LUMAAAAJ&hl=en) (IBM Research), [Guillermo Cecchi](https://researcher.watson.ibm.com/researcher/view.php?person=us-gcecchi) (IBM Research).





For the latest full paper: https://arxiv.org/abs/2010.11413



All the experimental results can be reproduced using the code in this repository. Feel free to contact me by doerlbh@gmail.com if you have any question about our work.



**Abstract**

Unlike traditional time series, the action sequences of human decision making usually involve many cognitive processes such as beliefs, desires, intentions and theory of mind, i.e. what others are thinking. This makes predicting human decision making challenging to be treated agnostically to the underlying psychological mechanisms. We propose to use a recurrent neural network architecture based on long short-term memory networks (LSTM) to predict the time series of the actions taken by the human subjects at each step of their decision making, the first application of such methods in this research domain. In this study, we collate the human data from 8 published literature of the Iterated Prisoner's Dilemma comprising 168,386 individual decisions and postprocess them into 8,257 behavioral trajectories of 9 actions each for both players. Similarly, we collate 617 trajectories of 95 actions from 10 different published studies of Iowa Gambling Task experiments with healthy human subjects. We train our prediction networks on the behavioral data from these published psychological experiments of human decision making, and demonstrate a clear advantage over the state-of-the-art methods in predicting human decision making trajectories in both single-agent scenarios such as the Iowa Gambling Task and multi-agent scenarios such as the Iterated Prisoner's Dilemma. In the prediction, we observe that the weights of the top performers tends to have a wider distribution, and a bigger bias in the LSTM networks, which suggests possible interpretations for the distribution of strategies adopted by each group. 


## Info

Language: Python3


Platform: MacOS, Linux, Windows

by Baihan Lin, August 2020

  

  


## Citation

If you find this work helpful, please try the models out and cite our works. Thanks!

    @article{lin2020predicting,
      title={Predicting Human Decision Making in Psychological Tasks with Recurrent Neural Networks},
      author={Lin, Baihan and Bouneffouf, Djallel and Cecchi, Guillermo},
      journal={arXiv preprint arXiv:2010.11413},
      year={2020}
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



