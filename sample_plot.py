from calculate_distances import *
import matplotlib.pyplot as plt

def plot_useen_clients_cdf(file_name,gammas = {"chi":[0.04],"kl":[0.2]},delta = 0.05, seen_clients_num = 100): 
    data_file = pd.read_csv(file_name)
    accuracies = data_file['accuracy'].values
    losses = data_file['loss'].values
    sample_losses = losses[:seen_clients_num]
    sample_accuracies = accuracies[:seen_clients_num]
    for dis in ["chi","kl"]:
        for gamma in gammas[dis]:
            n = len(accuracies)
            if(dis == "chi"):
                weights,distance = calculate_chisquare_weights_and_distance(sample_accuracies,sample_losses,gamma)
            else:
                weights,distance = calculate_kl_weights_and_distance(sample_accuracies,sample_losses,gamma)
        ##f_divergence
        x,y = calculate_cdf(weights)
        plt.step(x,y, linestyle='--',label=f'$d_{{{dis}}}={round(distance, 3)}$',linewidth=2.4)
        y_new = calculate_dkw_bound(y,delta)
        plt.fill_between(x, y, y_new, alpha=0.3)
    ##sample
    n = len(sample_accuracies)
    x,y = calculate_cdf([(x,1/n) for x in  sample_accuracies])
    plt.step(x,y, linestyle='--',label=f'$Meta_{{{n}}}$',linewidth=2.4)
    y_new = calculate_dkw_bound(y,delta)
    plt.fill_between(x, y, y_new, alpha=0.3)
    ##real
    n = len(accuracies)
    x,y = calculate_cdf([(x,1/n) for x in  accuracies])
    plt.step(x,y, linestyle='-',label=f'$Meta$',linewidth=2.4)
    plt.xlabel('accuracies')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.show()

def plot_meta_fdivergence(file_name1,file_name2,dist_function='chi',gammas=[0.05,0.02],delta=0.05):
    meta1_file = pd.read_csv(file_name1)
    meta1_accs,meta1_loss = meta1_file['accuracy'].values,meta1_file['loss'].values
    meta2_file = pd.read_csv(file_name2)
    meta2_accs = meta2_file['accuracy'].values
    plt.figure(figsize=(7,3))
    n = len(meta1_accs)
    for meta in ['Source','Target']:
        if(meta == 'Source'):
            acc_data = meta1_accs
        if(meta == 'Target'):
            acc_data = meta2_accs
        n = len(acc_data)
        ##meta networks
        x,y = calculate_cdf([(x,1/n) for x in  acc_data])
        plt.step(x,y, linestyle='-',label=f'${{{meta}}}$',linewidth=2)
        y_new = calculate_dkw_bound(y,delta)
        plt.fill_between(x, y, y_new, alpha=0.3)
    ##f-divergence network
    for gamma in gammas:
        if(dist_function == "chi"):
            weights,distance = calculate_chisquare_weights_and_distance(meta1_accs,meta1_loss,gamma)
        else:
            weights,distance = calculate_kl_weights_and_distance(meta1_accs,meta1_loss,gamma)
        x,y = calculate_cdf(weights)
        plt.step(x,y, linestyle='--',label=f'$d_{{{dist_function}}}={round(distance, 3)}$',linewidth=2.4)
        y_new = calculate_dkw_bound(y,delta)
        plt.fill_between(x, y, y_new, alpha=0.3)
    ## calculate two meta-distribution distance
    if(dist_function == 'chi'):
        div = calculate_meta_chisquare_divergence(meta2_accs, meta1_accs,10)
    if(dist_function == 'kl'):
        div = calculate_meta_kl_divergence(meta2_accs, meta1_accs, bins=10)
    plt.xlabel('Accuracy')
    plt.ylabel('Cumulative Probability')
    plt.title(f'divergence between Source and Target Meta Distributions: {div:.3f}')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    unseen_client_file_name = 'YOUR CDF CSV FILE'
    plot_useen_clients_cdf(unseen_client_file_name)
    meta1_file_name = 'YOUR CDF CSV FILE'
    meta2_file_name = 'YOUR CDF CSV FILE'
    plot_meta_fdivergence(meta1_file_name,meta2_file_name)