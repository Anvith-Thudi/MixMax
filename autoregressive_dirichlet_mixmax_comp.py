import numpy as np
import jax
import jax.numpy as jnp
import math
import os

import argparse

import scipy

import matplotlib.pyplot as plt

from transformers import GPTNeoXForCausalLM, GPTNeoXConfig
import torch
import torch.nn as nn

def generate_data(states, transition, length):

  #will start at initial always (should not matter for long sequences)
  sequence = [states[0]]

  for i in range(length):

    new = np.random.choice(a = states, p = transition[sequence[-1]])
    sequence.append(new)

  return sequence


def generate_data_stationary(states, transition, stationary, length):

  #will start at initial always (should not matter for long sequences)

  first = np.random.choice(a = states, p = stationary)
  sequence = [first]

  for i in range(length):

    new = np.random.choice(a = states, p = transition[sequence[-1]])
    sequence.append(new)

  return sequence

def get_samples(transition, stationary, states, max_length, samples_per_length):

  samples = []

  for i in range(0,max_length):

    i_samples = []

    for n in range(samples_per_length):
        i_samples.append(generate_data_stationary(states, transition,stationary, i))

    samples.append(i_samples)

  return samples


def stationary(transition):

  transition_t = np.transpose(transition)
  eigenvals, eigenvects = np.linalg.eig(transition_t)
  indx = np.isclose(eigenvals,1)

  stationary_dis = eigenvects[:,indx][:,0] / np.sum(eigenvects[:,indx][:,0])

  return stationary_dis.real


def transformer_prob_fn(model,sequence,stationary):
  probs = np.array([stationary[sequence[i,0]] for i in range(len(sequence))])



  for i in range(1,len(sequence[0])):

    input = torch.tensor(sequence[:,:i]).to(device)

    out = model(input)

    logits = out.logits
    preds = logits[:,-1,:]

    next_prob_all = nn.functional.softmax(preds, dim =1)

    next_prob = np.array([next_prob_all[j,sequence[j,i]].detach().cpu().item() for j in range(len(sequence))])

    probs = np.multiply(probs, next_prob)

  return probs

def transformer_f_lambda(params, ref_models, stationaries, sequence):

  probs = jnp.array([params[i]*transformer_prob_fn(ref_models[i], sequence, stationaries[i]) for i in range(len(ref_models))])

  return jnp.sum(probs, axis = 0)

def transformer_loss_fn(params, ref_models, stationaries, samples, max_length):

  losses = []

  for i in range(0,max_length):


    loss_i = []

    for j in range(len(samples)):

      j_samples = samples[j][i]


      out_probs = transformer_f_lambda(params, ref_models,stationaries,np.array(j_samples)) 
      neg_log_probs = (-1.0)* jnp.log(out_probs)

      exp_CE_loss = jnp.mean(neg_log_probs)

      loss_i.append(exp_CE_loss)


    weighted_loss_i = jnp.dot(params, jnp.array(loss_i))
    losses.append(weighted_loss_i)

  return (-1.0)*jnp.mean(jnp.array(losses))


def eval_transformer_loss_fn(params, ref_models, stationaries, samples, max_length):

  losses = []

  for i in range(0,max_length):

    loss_i = []

    for j in range(len(samples)):

      j_samples = samples[j][i]


      out_probs = transformer_f_lambda(params, ref_models,stationaries,np.array(j_samples)) 
      neg_log_probs = (-1.0)* jnp.log(out_probs)

      exp_CE_loss = jnp.mean(neg_log_probs)

      loss_i.append(exp_CE_loss)

    losses.append(loss_i)


  return jnp.mean(jnp.array(losses), axis = 0)

def transformer_update(params, ref_models, stationaries, samples, max_length, lr):

  grad = jax.grad(transformer_loss_fn)(params, ref_models, stationaries, samples, max_length)

  denominator = jnp.dot(params, jnp.exp(-lr*grad))


  return jnp.multiply(params, jnp.exp(-lr*grad)) / denominator

def get_batch(u, samples, b):


    dist_ind = np.random.choice(np.arange(len(u)), size = b, p = u)
    #inds = np.random.randint(low = 0, high = len(samples[0]), size = b)

    batch_samples = []

    dist_counts = [np.sum(dist_ind == i) for i in range(len(u))]

    #print(dist_counts)

    for i,count in enumerate(dist_counts):

        #dist_samples = []
        inds = np.random.randint(low = 0, high = len(samples[i]), size = count)
        dist_samples = [np.array(samples[i][j])[inds] for j in range(len(samples[i]))]
        dist_samples = list(dist_samples)
        #print(dist_samples.shape)

        batch_samples.append(dist_samples)

    return batch_samples


def prob_fn(sequence,transition,stationary):
  prob = stationary[sequence[0]]


  for i in range(1,len(sequence)):

    next_prob = transition[sequence[i-1]][sequence[i]]
    prob = prob*next_prob

  return prob


#NOTE: all the "fast" implementations are actually slower
def prob_fn_fast(sequence,transition,stationary):

  probs = [stationary[sequence[0]]]

  trans_t = jnp.array(np.transpose(transition))

  one_hot = np.zeros((sequence.size, len(transition[0])))
  one_hot[np.arange(sequence.size), sequence] = 1
  S = jnp.array(one_hot)


  Prob_matrix = jnp.matmul(trans_t, jnp.transpose(S))

  short_prob = jnp.transpose(Prob_matrix[:,:-1])
  short_S = S[1:,]

  for j in range(len(short_prob)):
    probs.append(jnp.dot(short_prob[j],short_S[j]))


  return jnp.prod(jnp.array(probs))


def f_lambda(params, transitions, stationaries, sequence):

  probs = [prob_fn(sequence, transitions[i], stationaries[i]) for i in range(len(transitions))]

  return jnp.dot(params, jnp.array(probs))

def f_lambda_fast(params, transitions, stationaries, sequence):

  probs = [prob_fn_fast(sequence, transitions[i], stationaries[i]) for i in range(len(transitions))]

  return jnp.dot(params, jnp.array(probs))


def get_samples(transition, stationary, states, max_length, samples_per_length):

  samples = []

  for i in range(0,max_length):

    i_samples = []

    for n in range(samples_per_length):
        i_samples.append(generate_data_stationary(states, transition,stationary, i))

    samples.append(i_samples)

  return samples




def loss_fn(params, transitions, stationaries, samples, max_length):

  losses = []

  for i in range(0,max_length):

    loss_i = []

    for j in range(len(samples)):

      j_samples = samples[j][i]


      out_probs = jnp.array([f_lambda(params, transitions,stationaries,sequence) for sequence in j_samples])
      neg_log_probs = (-1.0)* jnp.log(out_probs)

      exp_CE_loss = jnp.mean(neg_log_probs)

      loss_i.append(exp_CE_loss)


    weighted_loss_i = jnp.dot(params, jnp.array(loss_i))
    losses.append(weighted_loss_i)

  return (-1.0)*jnp.mean(jnp.array(losses))

def loss_fn_fast(params, transitions, stationaries, samples, max_length):

  losses = []

  for i in range(0,max_length):

    loss_i = []

    for j in range(len(samples)):

      j_samples = samples[j][i]


      out_probs = jnp.array([f_lambda_fast(params, transitions,stationaries,np.array(sequence)) for sequence in j_samples])
      neg_log_probs = (-1.0)* jnp.log(out_probs)

      exp_CE_loss = jnp.mean(neg_log_probs)

      loss_i.append(exp_CE_loss)


    weighted_loss_i = jnp.dot(params, jnp.array(loss_i))
    losses.append(weighted_loss_i)

  #return negative loss for maximization with grad descent
  return (-1.0)*jnp.mean(jnp.array(losses))


def eval_loss_fn(params, transitions, stationaries, samples, max_length):
  #returns loss on each distribution

  losses = []

  for i in range(0,max_length):

    loss_i = []

    for j in range(len(samples)):

      j_samples = samples[j][i]


      out_probs = jnp.array([f_lambda(params, transitions,stationaries,sequence) for sequence in j_samples])
      neg_log_probs = (-1.0)* jnp.log(out_probs)

      exp_CE_loss = jnp.mean(neg_log_probs)

      loss_i.append(exp_CE_loss)


    losses.append(loss_i)


  return jnp.mean(jnp.array(losses), axis = 0)

def update(params, transitions, stationaries, samples, max_length, lr):

  grad = jax.grad(loss_fn)(params, transitions, stationaries, samples, max_length)

  denominator = jnp.dot(params, jnp.exp(-lr*grad))


  return jnp.multiply(params, jnp.exp(-lr*grad)) / denominator


if __name__ == '__main__':
  
    parser = argparse.ArgumentParser(description='Settings')
    #parser.add_argument('--dataset', type = str, default= 'diabetes')
    parser.add_argument('--n_states', type = int, default= 4)
    parser.add_argument('--n_dists', type = int, default = 3)
    parser.add_argument('--mag', type = float, default = 1.0)
    parser.add_argument('--max_length', type = int, default = 10)
    parser.add_argument('--our_train_samples', type = int, default = 800)
    parser.add_argument('--our_eval_samples', type = int, default = 200)
    parser.add_argument('--our_steps', type = int, default = 10)
    parser.add_argument('--our_lr', type = float, default = 2.0)
    parser.add_argument('--ref_epochs', type = int, default = 20)
    parser.add_argument('--lr', type = float, default = 0.01)
    parser.add_argument('--batch_size', type = int, default = 200)
    parser.add_argument('--trial', type = int, default = 1)

    args = parser.parse_args()

    n_states = args.n_states
    n_dists = args.n_dists
    mag = args.mag
    max_length = args.max_length
    n_test_samples = args.our_eval_samples
    n_train_samples = args.our_train_samples
    num_steps = args.our_steps
    our_lr = args.our_lr
    n_epochs = args.ref_epochs
    batch_size = args.batch_size
    lr = args.lr


    trial = args.trial

    device = 'cuda:0'

    results_dir = f'DEFINE HERE'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if os.path.exists(results_dir + '/' + 'opt_losses.npy'):
       print("results already computed")
       exit()

    print("Creating Markov Chains")

    states = np.arange(n_states)

    transitions = []
    stationaries = []

    for i in range(n_dists):

        factors = mag*np.ones(n_states)
        normalized = np.random.dirichlet(factors, size = n_states)

        transitions.append(normalized)
        stationary_dist = stationary(normalized)
        stationaries.append(stationary_dist)

    np.save(results_dir + '/' + 'transitions', np.stack(transitions, axis =0))
    np.save(results_dir + '/' + 'stationaries', np.stack(stationaries, axis =0))


    print("Running Optimal With All the Samples")


    samples = [get_samples(transitions[i], stationaries[i], states, max_length, n_train_samples) for i in range(len(transitions))]

    opt_params = jnp.ones(len(samples))*(1/len(samples))

    loss_list = []
    opt_params_list = [opt_params]

    for t in range(num_steps):
        loss = loss_fn(opt_params, transitions, stationaries, samples, max_length)
        loss_list.append(loss)

        opt_params = update(opt_params, transitions, stationaries, samples, max_length, our_lr)
        opt_params_list.append(opt_params)

    np.save(results_dir + '/' + 'opt_params', np.array(opt_params_list[-1]))




    print("Running Biased (reusing All the Samples)")

    config = GPTNeoXConfig(vocab_size = len(states), hidden_size = 6, num_hidden_layers= 2, num_attention_heads = 2, 
                       intermediate_size = 8, max_position_embeddings= 12)

    CE_loss = nn.CrossEntropyLoss()

    train_samples = [get_samples(transitions[i], stationaries[i], states, max_length, n_train_samples) for i in range(len(transitions))]


    ref_models = []

    for i in range(len(transitions)):
        i_train_set = train_samples[i]

        model = GPTNeoXForCausalLM(config)

        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

        lengths = np.arange(1,max_length)

        for e in range(n_epochs):

            permute_sizes = np.random.permutation(lengths)

            for size in permute_sizes:

                data = torch.tensor(i_train_set[size]).to(device)

                input = data[:,:size]
                target = data[:, size]

                out = model(input)
                logits = out.logits

                pred = logits[:,-1,:]
                loss = CE_loss(pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        ref_models.append(model)

    #Our Weights
  
    biased_params = jnp.ones(len(train_samples))*(1/len(train_samples))

    biased_loss_list = []
    biased_params_list = [biased_params]


    for t in range(num_steps):
        loss = transformer_loss_fn(biased_params, ref_models, stationaries, train_samples, max_length)
        biased_loss_list.append(loss)

        biased_params = transformer_update(biased_params, ref_models, stationaries, train_samples, max_length, our_lr)
        biased_params_list.append(biased_params)

    np.save(results_dir + '/' + 'biased_params', np.array(biased_params_list[-1]))

    biased_params_t = torch.from_numpy(np.array(biased_params_list[-1]))


    print("Now Running 25-75 split")

    ref_train_size = round(0.25*n_train_samples)
    train_samples = [get_samples(transitions[i], stationaries[i], states, max_length, ref_train_size) for i in range(len(transitions))]

    config = GPTNeoXConfig(vocab_size = len(states), hidden_size = 6, num_hidden_layers= 2, num_attention_heads = 2, 
                       intermediate_size = 8, max_position_embeddings= 12)

    CE_loss = nn.CrossEntropyLoss()

    ref_models = []

    for i in range(len(transitions)):
        i_train_set = train_samples[i]

        model = GPTNeoXForCausalLM(config)

        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

        lengths = np.arange(1,max_length)

        for e in range(n_epochs):

            permute_sizes = np.random.permutation(lengths)

            for size in permute_sizes:

                data = torch.tensor(i_train_set[size]).to(device)

                input = data[:,:size]
                target = data[:, size]

                out = model(input)
                logits = out.logits

                pred = logits[:,-1,:]
                loss = CE_loss(pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        ref_models.append(model)

    #Our Weights
    mixmax_samples_per_length = round(0.75*n_train_samples)
    mixmax_samples = [get_samples(transitions[i], stationaries[i], states, max_length, mixmax_samples_per_length) for i in range(len(transitions))]

    params_75 = jnp.ones(len(mixmax_samples))*(1/len(mixmax_samples))

    split_75_loss_list = []
    split_75_params_list = [params_75]


    for t in range(num_steps):
        loss = transformer_loss_fn(params_75, ref_models, stationaries, mixmax_samples, max_length)
        split_75_loss_list.append(loss)

        params_75 = transformer_update(params_75, ref_models, stationaries, mixmax_samples, max_length, our_lr)
        split_75_params_list.append(params_75)

    np.save(results_dir + '/' + 'params_75', np.array(split_75_params_list[-1]))

    params_75_t = torch.from_numpy(np.array(split_75_params_list[-1]))


    print("Now Running 50-50 split")

    ref_train_size = round(0.5*n_train_samples)
    train_samples = [get_samples(transitions[i], stationaries[i], states, max_length, ref_train_size) for i in range(len(transitions))]

    config = GPTNeoXConfig(vocab_size = len(states), hidden_size = 6, num_hidden_layers= 2, num_attention_heads = 2, 
                       intermediate_size = 8, max_position_embeddings= 12)

    CE_loss = nn.CrossEntropyLoss()

    ref_models = []

    for i in range(len(transitions)):
        i_train_set = train_samples[i]

        model = GPTNeoXForCausalLM(config)

        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

        lengths = np.arange(1,max_length)

        for e in range(n_epochs):

            permute_sizes = np.random.permutation(lengths)

            for size in permute_sizes:

                data = torch.tensor(i_train_set[size]).to(device)

                input = data[:,:size]
                target = data[:, size]

                out = model(input)
                logits = out.logits

                pred = logits[:,-1,:]
                loss = CE_loss(pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        ref_models.append(model)

    #Our Weights
    mixmax_samples_per_length = round(0.5*n_train_samples)
    mixmax_samples = [get_samples(transitions[i], stationaries[i], states, max_length, mixmax_samples_per_length) for i in range(len(transitions))]

    params_5 = jnp.ones(len(mixmax_samples))*(1/len(mixmax_samples))

    split_5_loss_list = []
    split_5_params_list = [params_5]

    for t in range(num_steps):
        loss = transformer_loss_fn(params_5, ref_models, stationaries, mixmax_samples, max_length)
        split_5_loss_list.append(loss)

        params_5 = transformer_update(params_5, ref_models, stationaries, mixmax_samples, max_length, our_lr)
        split_5_params_list.append(params_5)

    np.save(results_dir + '/' + 'params_5', np.array(split_5_params_list[-1]))

    params_5_t = torch.from_numpy(np.array(split_5_params_list[-1]))


    print("Now Running 75-25 split")

    ref_train_size = round(0.75*n_train_samples)
    train_samples = [get_samples(transitions[i], stationaries[i], states, max_length, ref_train_size) for i in range(len(transitions))]

    config = GPTNeoXConfig(vocab_size = len(states), hidden_size = 6, num_hidden_layers= 2, num_attention_heads = 2, 
                       intermediate_size = 8, max_position_embeddings= 12)

    CE_loss = nn.CrossEntropyLoss()

    ref_models = []

    for i in range(len(transitions)):
        i_train_set = train_samples[i]

        model = GPTNeoXForCausalLM(config)

        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

        lengths = np.arange(1,max_length)

        for e in range(n_epochs):

            permute_sizes = np.random.permutation(lengths)

            for size in permute_sizes:

                data = torch.tensor(i_train_set[size]).to(device)

                input = data[:,:size]
                target = data[:, size]

                out = model(input)
                logits = out.logits

                pred = logits[:,-1,:]
                loss = CE_loss(pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        ref_models.append(model)

    #Our Weights
    mixmax_samples_per_length = round(0.25*n_train_samples)
    mixmax_samples = [get_samples(transitions[i], stationaries[i], states, max_length, mixmax_samples_per_length) for i in range(len(transitions))]

    params_25 = jnp.ones(len(mixmax_samples))*(1/len(mixmax_samples))

    split_25_loss_list = []
    split_25_params_list = [params_25]


    for t in range(num_steps):
        loss = transformer_loss_fn(params_25, ref_models, stationaries, mixmax_samples, max_length)
        split_25_loss_list.append(loss)

        params_25 = transformer_update(params_25, ref_models, stationaries, mixmax_samples, max_length, our_lr)
        split_25_params_list.append(params_25)

    np.save(results_dir + '/' + 'params_25', np.array(split_25_params_list[-1]))

    params_25_t = torch.from_numpy(np.array(split_25_params_list[-1]))



    print("Now collecting loss statistics")

    test_samples = [get_samples(transitions[i], stationaries[i], states, max_length, n_test_samples) for i in range(len(transitions))]

    opt_losses = eval_loss_fn(opt_params_list[-1], transitions, stationaries, test_samples, max_length)
    biased_losses = eval_loss_fn(biased_params_list[-1], transitions, stationaries, test_samples, max_length)
    split_75_losses = eval_loss_fn(split_75_params_list[-1], transitions, stationaries, test_samples, max_length)
    split_5_losses = eval_loss_fn(split_5_params_list[-1], transitions, stationaries, test_samples, max_length)
    split_25_losses = eval_loss_fn(split_25_params_list[-1], transitions, stationaries, test_samples, max_length)

    np.save(results_dir + '/' + 'opt_losses', np.array(opt_losses))
    np.save(results_dir + '/' + 'biased_losses', np.array(biased_losses))
    np.save(results_dir + '/' + 'split_75_losses', np.array(split_75_losses))
    np.save(results_dir + '/' + 'split_5_losses', np.array(split_5_losses))
    np.save(results_dir + '/' + 'split_25_losses', np.array(split_25_losses))






