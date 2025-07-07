import numpy as np
import pickle
import torch
torch.manual_seed(42)
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from hyperopt import hp, fmin, tpe, space_eval, Trials
from hyperopt.pyll.base import scope

# calculate log(1 + erf(x)) with a cutoff at x < 0.0
def log_erf(x):
    mask = (x < 0.0)
    bad_values = torch.zeros_like(x, device=x.device)
    x_under = torch.where(mask, x, bad_values)
    x_over = torch.where(~mask, x, bad_values)

    f_under = lambda x: torch.inf*torch.ones(size=x.size(), device=x.device)
    f_over = lambda x: torch.log(1.0 + torch.erf(x))

    return f_under(x_under) + f_over(x_over)

# custom loss function: MSE + erf(x) term
class Loss_Func(torch.nn.Module):
    def __init__(self, maxes):
        super(Loss_Func, self).__init__()
        self.maxes = maxes

    def forward(self, predictions, targets):
        return torch.mean(log_erf(self.maxes - predictions) + (predictions - targets)**2)

# pytorch MLP
class MLP(torch.nn.Module):
    # initialize pytorch MLP with specified number of input/hidden/output nodes
    def __init__(self, n_feature, n_hidden, n_output, num_hidden_layers, dropout_p):
        super(MLP, self).__init__()
        self.input = torch.nn.Linear(n_feature, n_hidden).to("cpu")
        self.predict = torch.nn.Linear(n_hidden, n_output).to("cpu")

        self.hiddens = []
        for i in range(num_hidden_layers-1):
            self.hiddens.append(torch.nn.Linear(n_hidden, n_hidden).to("cpu"))

        self.dropouts = []
        for i in range(num_hidden_layers):
            self.dropouts.append(torch.nn.Dropout(dropout_p).to("cpu"))

        # means and stds for inputs across the training set
        self.mass_means = np.array([-5.47074599, -5.50485362, -5.55107994])
        self.mass_stds = np.array([0.86575343, 0.86634857, 0.80166568])
        self.orb_means = np.array([1.10103604e+00, 1.34531896e+00, 1.25862804e+00, -1.44014696e+00,
                                   -1.44382469e+00, -1.48199204e+00, -1.58037780e+00, -1.59658646e+00,
                                   -1.46025210e+00, 5.35126034e-04, 2.93399827e-04, -2.07964769e-04,
                                   1.84826520e-04, 2.00942518e-04, 1.34561831e-03, -3.81075318e-02,
                                   -4.50480364e-02, -8.37049604e-02, 4.20809298e-02, 4.89242546e-02,
                                   7.81205381e-02])
        self.orb_stds = np.array([0.17770125, 0.27459303, 0.30934483, 0.60370379, 0.5976446, 0.59195887,
                                  0.68390679, 0.70470389, 0.62941292, 0.70706072, 0.70825354, 0.7082275,
                                  0.70715261, 0.70595807, 0.70598297, 0.68020376, 0.67983686, 0.66536654,
                                  0.73082135, 0.73034166, 0.73768424])
        
        # means and stds for outputs
        self.output_means = np.array([1.20088092, 1.31667089, -1.4599554, -1.16721504, -2.10491322, -1.3807749])
        self.output_stds = np.array([0.23123815, 0.63026354, 0.51926874, 0.49942197, 0.74455827, 0.58098256])
        self.output_maxes = torch.tensor((np.array([50.0, 50.0, 0.0, 0.0, np.log10(np.pi), np.log10(np.pi)]) - self.output_means)/self.output_stds, dtype=torch.float32).to("cpu")

        # save means and stds
        self.input_means = np.concatenate((self.mass_means, np.tile(self.orb_means, 100)))
        self.input_stds = np.concatenate((self.mass_stds, np.tile(self.orb_stds, 100)))

    # function to compute output pytorch tensors from input pytorch tensors
    def forward(self, x):
        x = torch.relu(self.input(x))
        x = self.dropouts[0](x)

        for i in range(len(self.hiddens)):
            x = torch.relu(self.hiddens[i](x))
            x = self.dropouts[i+1](x)

        x = self.predict(x)
        return x

    # function to get means and stds from time series inputs
    def get_means_stds(self, inputs, min_nt=5):
        masses = inputs[:,:3]
        orb_elements = inputs[:,3:].reshape((len(inputs), 100, 21))

        if self.training: # add statistical noise
            nt = np.random.randint(low=min_nt, high=101) # select nt randomly from 5-100
            rand_inds = np.random.choice(100, size=nt, replace=False) # choose snapshots without replacement
            means = torch.mean(orb_elements[:,rand_inds,:], dim=1)
            stds = torch.std(orb_elements[:,rand_inds,:], dim=1)

            rand_means = torch.normal(means, stds/(nt**0.5))
            rand_stds = torch.normal(stds, stds/((2*nt - 2)**0.5))
            pooled_inputs = torch.concatenate((masses, rand_means, rand_stds), dim=1)
        else: # no statistical noise
            means = torch.mean(orb_elements, dim=1)
            stds = torch.std(orb_elements, dim=1)
            pooled_inputs = torch.concatenate((masses, means, stds), dim=1)

        return pooled_inputs

    # training function
    def train_model(self, Xs, Ys, learning_rate=1e-3, weight_decay=0.0, epochs=1000, batch_size=2000, min_nt=5):
        # normalize inputs and outputs
        Xs = (Xs - self.input_means)/self.input_stds
        Ys = (Ys - self.output_means)/self.output_stds

        # split training data
        x_train, x_eval, y_train, y_eval = train_test_split(Xs, Ys, test_size=0.2, shuffle=False)
        x_train_var, y_train_var = torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
        x_validate, y_validate = torch.tensor(x_eval, dtype=torch.float32), torch.tensor(y_eval, dtype=torch.float32)

        # create Data Loader
        dataset = TensorDataset(x_train_var, y_train_var)
        train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True) #, num_workers=8)

        # loss function and optimizer
        loss_fn = Loss_Func(self.output_maxes)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # main training loop
        self.train()
        for i in range(epochs):
            for inputs, labels in train_loader:
                # clear gradient buffers
                optimizer.zero_grad()

                # get model predictions on training batch
                inputs  = inputs.to("cpu")
                pooled_inputs = self.get_means_stds(inputs, min_nt=min_nt)
                output = self(pooled_inputs)

                # get loss
                output, labels = output.to("cpu"), labels.to("cpu")
                loss = loss_fn(output, labels)
                
                # get gradients with respect to parameters
                loss.backward()

                # update parameters
                optimizer.step()

        # get loss on test set
        self.eval()
        x_validate, y_validate = x_validate.to("cpu"), y_validate.to("cpu")
        with torch.no_grad():
            pooled_x_validate = self.get_means_stds(x_validate, min_nt=min_nt)
            test_output = self(pooled_x_validate)
        test_loss = loss_fn(test_output, y_validate)

        return test_loss.item()
    
    # function to make predictions with trained model (takes and return numpy array)
    def make_pred(self, Xs):
        self.eval()
        Xs = (Xs - self.input_means)/self.input_stds
        pooled_Xs = self.get_means_stds(torch.tensor(Xs, dtype=torch.float32))
        Ys = self(pooled_Xs).detach().numpy()
        Ys = Ys*self.output_stds + self.output_means
        
        return Ys

# hidden parameters
def get_test_loss(args):
    n_hidden, num_hidden_layers, learning_rate, weight_decay, dropout_p, batch_size = args
    epochs = int(np.ceil(training_steps/np.ceil(int((0.8*len(training_inputs)))/batch_size))) #calculate number of epochs
    
    # initialize MLP
    mlp = MLP(n_feature=45, n_hidden=n_hidden, n_output=6, num_hidden_layers=num_hidden_layers, dropout_p=dropout_p)
    mlp = mlp.to("cpu")

    # train model
    test_loss = mlp.train_model(training_inputs, training_outputs, learning_rate=learning_rate, weight_decay=weight_decay, epochs=epochs, batch_size=batch_size, min_nt=5)

    return test_loss
    
if __name__ == "__main__":
    # load training set
    top_dir = '/scratch/gpfs/cl5968/'
    filename = top_dir + '/ML_data/regression_train_data_subset.pkl' # random 10% of training dataset
    f = open(filename, "rb")
    training_inputs = pickle.load(f)
    training_outputs = pickle.load(f)
    f.close()

    training_steps = 10_000 # fixed value

    space = [scope.int(hp.quniform('n_hidden', 10, 60, 5)),
             scope.int(hp.quniform('num_hidden_layers', 1, 6, 1)),
             hp.loguniform('learning_rate', -10.0, -1.0),
             hp.loguniform('weight_decay', -10.0, -1.0),
             hp.uniform('dropout_p', 0.0, 1.0),
             scope.int(hp.quniform('batch_size', 1000, 6_000, 1000))]

    trials = Trials()
    best = fmin(get_test_loss, space, algo=tpe.suggest, max_evals=100, trials=trials)
    
    print(best, '\n')
    print(space_eval(space, best), '\n')

    pickle.dump(trials, open(top_dir + "ML_models/hp_opt_trials_regressor.pkl", "wb"))
