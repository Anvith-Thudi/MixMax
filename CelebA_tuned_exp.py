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
    parser.add_argument('--sens_group', type = str, default= 'Male')
    parser.add_argument('--n_trees', type = int, default = 200)
    parser.add_argument('--depth', type = int, default = 8)
    parser.add_argument('--lr', type = float, default = 0.1)
    parser.add_argument('--allowed_feat', type = str, default = "all")
    parser.add_argument('--allowed_dist', type = str, default = 'all')
    parser.add_argument('--trial', type = int, default = 1)
    parser.add_argument('--num_steps', type = int, default = 20)
    parser.add_argument('--n_samples', type = int, default = 25000)
    parser.add_argument('--our_lr', type = float, default = 0.1)

    args = parser.parse_args()

    dataset = "CelebA_tuned"
    sensitive_group = args.sens_group
    n_trees = args.n_trees
    depth = args.depth
    lr = args.lr
    allowed_features = args.allowed_feat
    trial = args.trial
    allowed_distributions = args.allowed_dist
    num_steps = args.num_steps
    n_samples = args.n_samples
    our_lr = args.our_lr

    results_dir = f'DEFINE HERE'


    if not os.path.exists(results_dir):
        os.makedirs(results_dir)


    print("Processing Data")

    data_path = 'DEFINE HERE'
    celebA_df = pd.read_csv(data_path)

    cleaned_df = celebA_df.drop('image_id', axis = 1)
    more_clean_df = cleaned_df.drop(0, axis = 0)

    object_columns = cleaned_df.columns 
    category_df = cleaned_df.copy()

    for col in object_columns:
        category_df[col] = category_df[col].astype('category')

    final_df = category_df.copy()
    final_df[object_columns] = final_df[object_columns].apply(lambda x: x.cat.codes)

    labels = final_df['Attractive'].to_numpy()
    features = final_df.drop('Attractive', axis = 1).to_numpy()

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=.2)


    print("training individual models")


    if sensitive_group == 'Male':
        sens_feature = 19
        vals = np.arange(2)

    elif sensitive_group == 'Young':
        sens_feature = 38
        vals = np.arange(2)

    elif sensitive_group == 'Pale_Skin':
        sens_feature = 25
        vals = np.arange(2)


    if allowed_features == 'all':
        allowed_feature_inds = [i for i in range(len(features[0]))]

    elif allowed_features == 'ten':
        allowed_feature_inds = [i for i in range(10)]

    elif allowed_features == 'five':
        allowed_feature_inds = [i for i in range(5)]

    elif allowed_features == 'two':
        allowed_feature_inds = [i for i in range(2)]

    not_allowed_features = [19,38,25]

    if sens_feature not in not_allowed_features:
        not_allowed_features.append(sens_feature)

    for feat in not_allowed_features:
        if feat in allowed_feature_inds:
            allowed_feature_inds.remove(feat)

    allowed_feature_inds = np.array(allowed_feature_inds)


    sens_models = []
    sens_cov_dens = []
    sens_hists = []

    sens_train_sets = []
    sens_test_sets = []

    for val in vals:

        print(val)
        print(sens_feature)
        train_inds = train_features[:, sens_feature].astype(int) == val
        test_inds = test_features[:, sens_feature].astype(int) == val

        X_train = train_features[train_inds]
        y_train = train_labels[train_inds]

        print(X_train.shape)

        X_test = test_features[test_inds]
        y_test = test_labels[test_inds]

        print(allowed_feature_inds)

        X_train = X_train[:,allowed_feature_inds]
        X_test = X_test[:, allowed_feature_inds]

        sens_train_sets.append([X_train, y_train])
        sens_test_sets.append([X_test, y_test])

        print(X_train.shape)

        #using default settings in xgboost paper but with more estimators 
        bst = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, objective='binary:logistic')

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

    lr = our_lr

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







