import folktables 
import xgboost as xgb
import scipy
import numpy as np

import sklearn
import jax
import pandas as pd
import os

import argparse



from sklearn.model_selection import train_test_split
import sklearn.neighbors
import matplotlib.pyplot as plt
import jax.numpy as jnp

def CE_loss(out,y):

  
  b = np.zeros((len(y),2))

  #to handle if passed as boolean
  y = 1* y

  b[np.arange(len(y)),y] = 1

  return -np.log(np.sum(np.multiply(out,b), axis = 1))

def f_lambda_shift(params, optim_funcs, shift_funcs, inputs):


  og_probs = []
  probs = []
  shifts = []

  for i in range(len(params)):

    prob = optim_funcs[i].predict_proba(inputs)
    og_probs.append(params[i]*prob)


    shift_probs = np.exp(shift_funcs[i].score_samples(inputs))
    weighted_shift = params[i]*shift_probs

    weighted_prob = np.transpose(np.multiply(prob.T, weighted_shift))

    probs.append(weighted_prob)
    shifts.append(weighted_shift)

  denominator = np.sum(np.array(shifts), axis = 0)

  bad_inds = denominator == 0.0
    
  numerator = np.sum(np.array(probs), axis = 0)
  og_numerator = np.sum(np.array(og_probs), axis = 0)

  denominator[bad_inds] = 1.0
  numerator[bad_inds] = og_numerator[bad_inds]
  


  outs = np.transpose(np.divide(np.transpose(numerator), denominator))

  return outs

def losses_shift(params, optim_funcs, shift_funcs, samples, max_samples):

  losses = []

  for i in range(len(params)):

    max_ind = min(max_samples, len(samples[i]))

    out = f_lambda_shift(params, optim_funcs, shift_funcs, samples[i][0][:max_ind])
    labels = samples[i][1][:max_ind]
  
    loss = CE_loss(out, labels)

    cleaned_loss = loss[~np.isinf(loss)]
      
    losses.append(np.mean(cleaned_loss))

  
  return np.array(losses)

def eval_loss(model, samples):

  losses = []

  for i in range(len(samples)):
    
    out = model.predict_proba(samples[i][0])
    labels = samples[i][1]
  
    loss = CE_loss(out, labels)

    cleaned_loss = loss[~np.isinf(loss)]
      
    losses.append(np.mean(cleaned_loss))

  return losses
  

def update_shift(params, optim_funcs, shift_funcs, samples, max_samples, lr):

  #performing gradient ascent

  grad = losses_shift(params, optim_funcs, shift_funcs, samples, max_samples)

  denominator = jnp.dot(params, np.exp(lr*grad))

  return (np.multiply(params, np.exp(lr*grad)) / denominator, np.dot(grad, params))
    


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--dataset', type = str, default= 'income')
    parser.add_argument('--sens_group', type = str, default= 'sex')
    parser.add_argument('--n_trees', type = int, default = 200)
    parser.add_argument('--depth', type = int, default = 8)
    parser.add_argument('--lr', type = float, default = 0.1)
    parser.add_argument('--allowed_feat', type = str, default = "all")
    parser.add_argument('--allowed_dist', type = str, default = 'all')
    parser.add_argument('--trial', type = int, default = 1)

    args = parser.parse_args()


    states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
           'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
             'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']


    

    dataset = args.dataset
    sensitive_group = args.sens_group
    n_trees = args.n_trees
    depth = args.depth
    lr = args.lr
    allowed_features = args.allowed_feat
    trial = args.trial
    allowed_distributions = args.allowed_dist

    results_dir = f'DEFINE HERE'


    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    
    print("making dataset")

    data_source = folktables.ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    states_subsets = states[:10]


    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    if dataset == 'income':
        if allowed_features == 'all':
            allowed_feature_inds = np.array([i for i in range(10)])

        elif allowed_features == 'two':
            allowed_feature_inds = [0,2]

        elif allowed_features == 'one':
            allowed_feature_inds = [0]


        if sensitive_group == 'race':
            sens_feature = 9
            vals = np.arange(1,10)

        elif sensitive_group == 'sex':
            sens_feature = 8
            vals = np.arange(1,3)

    for state in states_subsets:

        print(state)
        data = data_source.get_data(states=[state], download=True)
    
        #FOR ACSIncome
        if dataset == "income":
            features, labels, _ = folktables.ACSIncome.df_to_numpy(data)
        


        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.2)


        train_features.append(X_train)
        train_labels.append(y_train)
        test_features.append(X_test)
        test_labels.append(y_test)


    concat_train_features = np.concatenate(train_features, axis = 0)
    concat_train_labels = np.concatenate(train_labels, axis = 0)
    concat_test_features = np.concatenate(test_features, axis = 0)
    concat_test_labels = np.concatenate(test_labels, axis = 0)

    print("now creating groups and individual models")

    sens_models = []
    sens_cov_dens = []
    sens_hists = []

    sens_train_sets = []
    sens_test_sets = []

    for val in vals:

        train_inds = concat_train_features[:, sens_feature].astype(int) == val
        test_inds = concat_test_features[:, sens_feature].astype(int) == val

        X_train = concat_train_features[train_inds]
        y_train = concat_train_labels[train_inds]

        X_test = concat_test_features[test_inds]
        y_test = concat_test_labels[test_inds]

        X_train = X_train[:,allowed_feature_inds]
        X_test = X_test[:, allowed_feature_inds]

        sens_train_sets.append([X_train, y_train])
        sens_test_sets.append([X_test, y_test])

        bst = xgb.XGBClassifier(n_estimators= n_trees, max_depth= depth, learning_rate= lr, objective='binary:logistic')

        # fit model
        bst.fit(X_train, y_train)   

        sens_models.append(bst)

        kde = sklearn.neighbors.KernelDensity(kernel = "gaussian", bandwidth = "scott").fit(X_train)

        sens_cov_dens.append(kde)

        if len(allowed_feature_inds) == 1:
            unique, counts = np.unique(X_train[:,0], return_counts=True)

            prob = counts / len(X_train[:,0])
            sens_hists.append(dict(zip(unique, prob)))



    print("taking the desired partition and evaluating")

    if allowed_distributions == 'all':
        allowed_dist = [i for i in range(len(sens_train_sets))]



    train_sets = []
    test_sets = []

    models = []
    cov_dens = []
    hists = []

    for i in allowed_dist:
        print(i)
        train_sets.append(sens_train_sets[i])
        test_sets.append(sens_test_sets[i])

        models.append(sens_models[i])
        cov_dens.append(sens_cov_dens[i])

        if len(sens_hists) > 0:
            hists.append(sens_hists[i])

    accuracies = []
    for i, model in enumerate(models):

        accs = []

        for j, test_set in enumerate(test_sets):
            acc = model.score(test_set[0], test_set[1])
            accs.append(acc)

        accuracies.append(accs)

    accuracies = np.array(accuracies)

    np.save(results_dir + '/' + 'ind_training_accs', accuracies)


    #an even mixture by upweighting to match n_samples from most common

    print("training even balancing baseline")

    train_features = [train_sets[i][0] for i in range(len(train_sets))]
    train_labels = [train_sets[i][1] for i in range(len(train_sets))]

    max_len = np.max(np.array([len(train_features[i]) for i in range(len(train_sets))]))
    weights = [(max_len / len(train_features[i]))*np.ones(len(train_features[i])) for i in range(len(train_sets))]


    features_np = np.concatenate(train_features, axis = 0)
    label_np = np.concatenate(train_labels, axis = 0)

    weights_np = np.concatenate(weights, axis = 0)

    #using default settings in og xgboost paper but with more estimators
    even_model = xgb.XGBClassifier(n_estimators=n_trees, max_depth=depth, learning_rate=lr, objective='binary:logistic')

    # fit model
    even_model.fit(features_np, label_np, sample_weight= weights_np) 

    accs = []

    for test_set in test_sets:
        acc = even_model.score(test_set[0], test_set[1])
        accs.append(acc)

    np.save(results_dir + '/' + 'even_training_accs', np.array(accs))


    #NOW our approach

    print("obtaining our weighting")

    n_samples = 25000
    num_steps = 20
    lr = 0.1

    params = jnp.ones(len(test_sets))*(1/len(test_sets))

    loss_list = []
    params_list = [params]

    for t in range(num_steps):

        params, loss = update_shift(params, models, cov_dens, train_sets, n_samples, lr)
        params_list.append(params)
        loss_list.append(loss)

    np.save(results_dir + '/' + 'optimization_losses', np.array(loss_list))
    np.save(results_dir + '/' + 'optimization_params', np.stack(params_list, axis = 0))


    print("training our model")

    final_params = params_list[-1]


    train_features = [train_sets[i][0] for i in range(len(train_sets))]
    train_labels = [train_sets[i][1] for i in range(len(train_sets))]

    lens = np.array([len(train_features[i]) for i in range(len(train_sets))])
    max_len = np.max(lens)
    arg_max_len = np.argmax(lens)

    #scaled so that dataset with most samples still uses all of it, and everything is scaled accordingly: note all have sample n_samples before final_params[i]

    #NOTE: by dividing by min I enforce we use more samples than prev, max ensures it's still less
    opt_weights = [(final_params[i]/np.max(final_params))*(max_len / len(train_features[i]))*np.ones(len(train_features[i])) for i in range(len(train_sets))]

    features_np = np.concatenate(train_features, axis = 0)
    label_np = np.concatenate(train_labels, axis = 0)

    opt_weights_np = np.concatenate(opt_weights, axis = 0)

    bst = xgb.XGBClassifier(n_estimators=n_trees, max_depth=depth, learning_rate=lr, objective='binary:logistic')
    # fit model
    bst.fit(features_np, label_np, sample_weight= opt_weights_np) 


    print("saving our accuracy and all losses")

    opt_accuracies = []
    for i in range(len(test_sets)):
        X_test, y_test = test_sets[i]
        opt_accuracy = bst.score(X_test, y_test)

        opt_accuracies.append(opt_accuracy)

    np.save(results_dir + '/' + 'optima_accs', np.array(opt_accuracies))

    opt_losses = eval_loss(bst, test_sets)
    even_losses = eval_loss(even_model, test_sets)

    np.save(results_dir + '/' + 'optima_losses', np.array(opt_losses))
    np.save(results_dir + '/' + 'even_losses', np.array(even_losses))







