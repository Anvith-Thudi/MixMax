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

  #will start at initial always 
  sequence = [states[0]]

  for i in range(length):

    new = np.random.choice(a = states, p = transition[sequence[-1]])
    sequence.append(new)

  return sequence


def generate_data_stationary(states, transition, stationary, length):

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
  #loss is average of losses over lengths from 1 to max_length (represent reading a document of lengths at most lax_length samples from the distribution)

  losses = []

  for i in range(0,max_length):

    loss_i = []

    for j in range(len(samples)):

      j_samples = samples[j][i]


      out_probs = transformer_f_lambda(params, ref_models,stationaries,np.array(j_samples)) #for sequence in j_samples])
      neg_log_probs = (-1.0)* jnp.log(out_probs)

      exp_CE_loss = jnp.mean(neg_log_probs)

      loss_i.append(exp_CE_loss)


    weighted_loss_i = jnp.dot(params, jnp.array(loss_i))
    losses.append(weighted_loss_i)

  #return negative loss for maximization with grad descent
  return (-1.0)*jnp.mean(jnp.array(losses))


def eval_transformer_loss_fn(params, ref_models, stationaries, samples, max_length):
  #returns loss on each distribution

  losses = []

  for i in range(0,max_length):

    loss_i = []

    for j in range(len(samples)):

      j_samples = samples[j][i]


      out_probs = transformer_f_lambda(params, ref_models,stationaries,np.array(j_samples)) #for sequence in j_samples])
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
    batch_samples = []

    dist_counts = [np.sum(dist_ind == i) for i in range(len(u))]

    for i,count in enumerate(dist_counts):

        #dist_samples = []
        inds = np.random.randint(low = 0, high = len(samples[i]), size = count)
        dist_samples = [np.array(samples[i][j])[inds] for j in range(len(samples[i]))]
        dist_samples = list(dist_samples)
        #print(dist_samples.shape)

        batch_samples.append(dist_samples)

    return batch_samples

#helper functions for optimal soln
# HELPER FUNCTIONS


def prob_fn(sequence,transition,stationary):
  prob = stationary[sequence[0]]

  for i in range(1,len(sequence)):

    next_prob = transition[sequence[i-1]][sequence[i]]
    prob = prob*next_prob

  return prob



def f_lambda(params, transitions, stationaries, sequence):

  probs = [prob_fn(sequence, transitions[i], stationaries[i]) for i in range(len(transitions))]

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
  #loss is average of losses over lengths from 1 to max_length (represent reading a document of lengths at most lax_length samples from the distribution)

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
    parser.add_argument('--n_states', type = int, default= 4)
    parser.add_argument('--n_dists', type = int, default = 3)
    parser.add_argument('--mag', type = float, default = 1.0)
    parser.add_argument('--max_length', type = int, default = 10)
    parser.add_argument('--our_train_samples', type = int, default = 300)
    parser.add_argument('--our_eval_samples', type = int, default = 200)
    parser.add_argument('--our_steps', type = int, default = 10)
    parser.add_argument('--our_lr', type = float, default = 2.0)
    parser.add_argument('--ref_epochs', type = int, default = 20)
    parser.add_argument('--lr', type = float, default = 0.01)
    parser.add_argument('--DRO_lr', type = float, default = 0.1)
    parser.add_argument('--n_steps', type = int, default = 150)
    parser.add_argument('--batch_size', type = int, default = 200)
    parser.add_argument('--trial', type = int, default = 1)

    args = parser.parse_args()

    n_states = args.n_states
    n_dists = args.n_dists
    mag = args.mag
    max_length = args.max_length
    opt_samples_per_length = args.our_eval_samples
    num_steps = args.our_steps
    our_lr = args.our_lr
    n_epochs = args.ref_epochs
    lr = args.lr
    alpha_lr = args.DRO_lr
    n_steps = args.n_steps
    batch_size = args.batch_size
    c = 1e-3

    trial = args.trial

    device = 'cuda:0'

    results_dir = f'DEFINE HERE'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)


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


    print("Obtaining Our Weights Using Transformers")

    config = GPTNeoXConfig(vocab_size = len(states), hidden_size = 6, num_hidden_layers= 2, num_attention_heads = 2, 
                       intermediate_size = 8, max_position_embeddings= 12)

    CE_loss = nn.CrossEntropyLoss()

    samples_per_length = args.our_train_samples + args.our_eval_samples
    train_samples = [get_samples(transitions[i], stationaries[i], states, max_length, samples_per_length) for i in range(len(transitions))]


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
  
    params = jnp.ones(len(train_samples))*(1/len(train_samples))

    transformer_loss_list = []
    transformer_params_list = [params]

    for t in range(num_steps):
        loss = transformer_loss_fn(params, ref_models, stationaries, train_samples, max_length)
        transformer_loss_list.append(loss)

        params = transformer_update(params, ref_models, stationaries, train_samples, max_length, our_lr)
        transformer_params_list.append(params)

    np.save(results_dir + '/' + 'our_params', np.array(transformer_params_list[-1]))

    our_params = torch.from_numpy(np.array(transformer_params_list[-1]))



    print("Now collecting loss statistics")

    test_samples_per_length = args.our_eval_samples
    test_samples = [get_samples(transitions[i], stationaries[i], states, max_length, test_samples_per_length) for i in range(len(transitions))]


    transformer_ours_losses = eval_loss_fn(transformer_params_list[-1], transitions, stationaries, test_samples, max_length)

    np.save(results_dir + '/' + 'transformer_ours_losses', np.array(transformer_ours_losses))

    our_ref_loss = eval_transformer_loss_fn(transformer_params_list[-1], ref_models, stationaries, test_samples, max_length)

    np.save(results_dir + '/' + 'our_losses', np.array(our_ref_loss))

    
    

    



    





    

    



    