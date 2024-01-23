import argparse
import HPO.gwnHPO as gwnHPO

import Train.gwnTrain as gwnTrain

import Evaluation.gwnEval as gwnEval

parser = argparse.ArgumentParser()

# Random Search HPO arguments
parser.add_argument('--num_configs', type=int, default=30, help='number of random configurations to search through')
parser.add_argument('--tune_gwn', type=bool, help='whether to perform random search HPO on GWN model')

# Train final GNN model arguments
parser.add_argument('--train_gwn', type=bool, help='whether to train final GWN model')

# Calculate metrics of final models' results arguments
parser.add_argument('--eval_gwn', type=bool, help='whether to report final gwn metrics')

parser.add_argument('--n_stations', type=int, default=27, help='number of weather stations')
# For n_split, total points = 28(0-27), set default to 26 so that the last 2 points (26 and 27) are used for validation and test
parser.add_argument('--n_split', type=int, default=26, help='number of splits in walk-forward validation')

# Graph WaveNet arguments, default arguments are optimal hyper-parameters
parser.add_argument('--device', type=str, default='cuda', help='device to place model on')
parser.add_argument('--data', type=str, default='Data/graph_stations.csv', help='weather stations data path')
parser.add_argument('--adjdata', type=str, default='Data/adj_matrix.pkl', help='weather stations adjacency matrix')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--aptonly', type=bool, default=True, help='whether only adaptive adj')
parser.add_argument('--addaptadj', type=bool, default=True, help='whether add adaptive adj')
parser.add_argument('--randomadj', type=bool, default=True, help='whether random initialize adaptive adj')
parser.add_argument('--gcn_bool', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--seq_length', type=int, default=24, help='length of output sequence') # was 24
parser.add_argument('--lag_length', type=int, default=24, help='length of input sequence') # was 12
parser.add_argument('--nhid', type=int, default=42, help='') # was 32
parser.add_argument('--in_dim', type=int, default=6, help='number of features in input')
parser.add_argument('--num_nodes', type=int, default=27, help='number of nodes in graph(num weather stations)')
parser.add_argument('--num_layers', type=int, default=4, help='number of layers') # was 2
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--patience', type=int, default=8, help='patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate') # was 0.2
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=60, help='') # was 30
parser.add_argument('--save', type=str, default='Garage/Final Models/GWN/', help='save path')

# Add arguments above to args for 
args = parser.parse_args()

if __name__ == '__main__':

    # list of all weather stations (27)
    stations = ['ALEXANDERBAAI', 'BEAUFORT-WES', 'BLOEMFONTEIN WO', 'CALVINIA WO', 'CAPE TOWN WO', 'CRADOCK-MUN', 
               'EAST LONDON WO', 'GARIEP DAM', 'GEORGE WO', 'GREYTOWN', 'KIMBERLEY WO', 'KING SHAKA INTERNATIONAL AIRPORT WO', 
               'KNELLPOORTDAM', 'KOKSTAD', 'KROONSTAD', 'LADYSMITH', 'MOOI RIVER', 'PORT ELIZABETH AWOS', 'PRINS ALBERT-SWARTRIVIER', 
               'QUEENSTOWN', 'REDELINGSHUYS-AWS', 'SUTHERLAND', 'UMTHATHA WO', 'UPINGTON WO', 'VAN ZYLSRUS', 'WARDEN-HERITAGE', 
               'WORCESTER-AWS']

    """
    List of data points for walk-forward validation with the successive and overlapping training-validation-test partition:
    The first point, 226872 is one year's worth of data (from 2012-01-01 00:00:00 to 2012-12-31 23:00:00), the next point is an
    increment value of 72695, which is 4 months worth of data. This results in rolling walk-forward validation where the train
    size increases each increment, with the validation and test sets each being 4 months' worth of data. Every point should be
    divisible by 27, the number of weather stations for sliding window operation.
    """

    # increment - Walk-forward validation split points.
    # increment = [226872, 299567, 372262, 444957, 517652, 590347, 663042, 735737, 808432, 881127, 
    #              953822, 1026517, 1099212, 1171907, 1244602, 1317297, 1389992, 1462687, 1535382, 
    #              1608077, 1680772, 1753467, 1826162, 1898857, 1971552, 2044247, 2116942, 2189558]
    
    increment = [226854, 299565, 372249, 444933, 517644, 590328, 663039, 735723, 808407, 881118, 
                 953802, 1026513, 1099197, 1171881, 1244592, 1317276, 1389987, 1462671, 1535382, 
                 1608066, 1680750, 1753461, 1826145, 1898856, 1971540, 2044224, 2116935, 2189538]


    # Random search GWN
    if args.tune_gwn:
        gwnHPO.hpo(increment, args)

    # Train final GWN models
    if args.train_gwn:
        gwnTrain.train(increment, args)

    # Record metrics for final GWN models
    if args.eval_gwn:
        gwnEval.eval(stations, args)

