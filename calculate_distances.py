import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import rel_entr

def calculate_dkw_epsilon(delta, n):
    log_value = np.log(n/delta)
    epsilon = np.sqrt(log_value / (2 * n))
    return epsilon

def check_dkw_oneside_constraint(epsilon,n):
    log_value = np.log(2)
    expression_value = (1 / (2 * n)) * log_value
    if(epsilon >= np.sqrt(expression_value)):
       return True
    else:
       return False
    
def calculate_dkw_bound(y_fn,delta):
    n = len(y_fn)
    epsilon = calculate_dkw_epsilon(delta,n)
    if(check_dkw_oneside_constraint(epsilon,n)):
        y_f = y_fn + epsilon
        y_f = np.clip(y_f, 0, 1)
    else:
        print(f'error: dkw constraint')
    return y_f
   
def chisquare_objective(lambdas, error, gamma, n):
    sum1 = np.mean(lambdas * error)
    sum2 = np.sum((lambdas - 1/n)**2)
    return -(sum1 - gamma * sum2)

chisquare_constraints = {
    'type': 'eq',
    'fun': lambda a: np.sum(a) - 1
}

def calculate_chisquare_weights_and_distance(accuracies,losses,gamma):
    client_num = len(accuracies)
    initial_ans = np.ones(client_num) / client_num
    bounds = [(0, 1) for _ in range(client_num)]
    result = minimize(chisquare_objective, initial_ans, args=(losses, gamma, client_num), method='SLSQP', bounds=bounds, constraints=chisquare_constraints)
    optimal_ans = result.x
    weights = []
    for i,a in enumerate(optimal_ans):
        weights.append((accuracies[i],a))
    distance = np.sum((optimal_ans - 1/client_num)**2)/(1/client_num)
    return weights,distance

def calculate_kl_weights_and_distance(accuracies,losses,gamma):
    client_num = len(accuracies)
    p = np.exp(losses/gamma)
    sum = np.sum(p)
    optimal_ans = p/sum
    weights = []
    for i,a in enumerate(optimal_ans):
        weights.append((accuracies[i],a))
    distance = np.sum(optimal_ans*np.log(client_num*optimal_ans))
    return weights,distance

def calculate_cdf(weights):
    sorted_arr = sorted(weights, key=lambda x: x[0])
    x,y = [],[]
    for i,t in enumerate(sorted_arr):
      x.append(t[0])
      if(i==0):
          y.append(t[1])
      else:
          y.append(t[1]+y[i-1])
    return x,y

def calculate_meta_chisquare_divergence(samples1, samples2, num_bins=10):
    hist1, bin_edges = np.histogram(samples1, bins=num_bins, range=(min(samples1.min(), samples2.min()), max(samples1.max(), samples2.max())), density=True)
    hist2, _ = np.histogram(samples2, bins=bin_edges, density=True)
    dx = np.diff(bin_edges)
    epsilon = min([x for x in hist2 if x!=0])
    hist2 = np.maximum(hist2, epsilon)
    chi_square = np.sum((hist1 - hist2) ** 2 / hist2 * dx)
    return chi_square

def calculate_meta_kl_divergence(samples1, samples2, bins=10):
    hist1, bin_edges = np.histogram(samples1, bins=bins,range=(min(samples1.min(), samples2.min()), max(samples1.max(), samples2.max())), density=True)
    hist2, _ = np.histogram(samples2, bins=bin_edges,density=True)
    dx = np.diff(bin_edges)
    epsilon = min([x for x in hist2 if x!=0])
    hist2 = np.maximum(hist2, epsilon)
    kl_div = np.sum(rel_entr(hist1, hist2) * dx)
    return kl_div