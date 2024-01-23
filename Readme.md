# Graph WaveNet for weather prediction

*Graph WaveNet pipeline for weather prediction using STGNN.* 


# Requirements

python3

See `requirements.txt`


# Installation

`virtualenv -p /usr/bin/python3.8 venv`

`source venv/bin/activate`

`pip3 install -r requirements.txt`


# Experiments

## Random-Search Hyper-Parameter Optimisation(HPO)

GWN HPO on 24 hour forecasting horizon:

`python3 main.py --tune_gwn=True`


## Training Models Using Optimal Hyper-Parameters

GWN GNN HPO on [1, 3, 6, 12, 24, 48] hour forecasting horizon:

`python3 main.py --train_gwn=True`


## Evaluate Models' Performance(MSE, RMSE, MAE, SMAPE)

GWN GNN HPO on [1, 3, 6, 12, 24, 48] hour forecasting horizon on each of the 21 weather stations:

`python3 main.py --eval_gwn=True`
