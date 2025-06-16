import h5py 
import numpy as np  
import torch

from tqdm import trange
from itertools import product

def generate_binary_code(bit_length, batch_size_exp=None, batch_number=0):
    # No batch size is given, all data is returned
    if batch_size_exp is None:
        batch_size_exp = bit_length
    batch_size = 2 ** batch_size_exp
    # Generate batch
    bit_combinations = np.zeros((batch_size, bit_length))
    for number in range(batch_size):
        dividend = number + batch_number * batch_size
        bit_index = 0
        while dividend != 0:
            bit_combinations[number, bit_index] = np.remainder(dividend, 2)
            dividend = np.floor_divide(dividend, 2)
            bit_index += 1
    return bit_combinations


def generate_bars_and_stripes_complete(length):
    """ Creates a dataset containing all possible samples showing bars or stripes.

    :param length: Length of the bars/stripes.
    :type length: int

    :return: Samples.
    :rtype: numpy array [num_samples, length*length]
    """
    stripes = generate_binary_code(length)
    stripes = np.repeat(stripes, length, 0)
    stripes = stripes.reshape(2 ** length, length * length)

    bars = generate_binary_code(length)
    bars = bars.reshape(2 ** length * length, 1)
    bars = np.repeat(bars, length, 1)
    bars = bars.reshape(2 ** length, length * length)
    return np.vstack((stripes[0:stripes.shape[0]-1],bars[1:bars.shape[0]]))
    # return numx.vstack((stripes, bars)) # Tests have to match if changed to this.


def sigmoid(x):
    """ Calcule et evalue la fonction sigmoid au point x. """
    return 1 / (1 + torch.exp(-x))


def all_possible_config(w, theta, eta, combinaisons):
    """ Calcule le log du denominateur de l'expression de la log-vraisemblance
        la somme sur toutes les possibilités de v,h (dans notre cas {0,1})
        renvoie un vecteur de taille Nv + Nh  """
    Nv, _ = w.shape

    parties_v = combinaisons[:, :Nv]  # ttes les combinaisons en gardant que les Nv premiers elt.
    parties_h = combinaisons[:, Nv:]  # idem pour les Nh derniers elt.

    # H prend en entrée une configuration possible
    # on sommera donc sur ttes les configurations possibles
    return -H(parties_v, parties_h, w, theta, eta)


def H(mat_v, mat_h, w, theta, eta):
    """
    Fonction d'energie du RBM 
        w - matrice de poids visible -> poids cache (Nv x Nh)
        theta - vecteur de taille Nv  
        eta - vecteur de taille Nh

        v : vec de tt les visibles  (taille Nv)
        h : vec de tt les caches    (taille Nh)
    """
    return -(torch.einsum("ki,ia,ka->k", mat_v, w, mat_h) + mat_v @ theta + mat_h @ eta)


def log_likelihood(mat_v, w, theta, eta, combinaisons):
    """ Calcule la log-vraisemblance selon la fonction de masse d'une RBM.
        renvoie un vecteur de taille Nh + Nv """

    #   log( 1/Z(w, theta, eta) * np.exp(-H(v, h, w, theta, eta)) )
    # = log( 1/Z(w, theta, eta) ) + log (np.exp(-H(v, h, w, theta, eta)))
    # = -log( Z(w, theta, eta) ) - H(v, h, w, theta, eta)
    # = -log ( sum([np.exp(-H(v, h, w, theta, eta)) for all (v, h)] ) - H(v, h, w, theta, eta)

    log_Z = torch.logsumexp(all_possible_config(w, theta, eta, combinaisons), dim=0)
    Ns = mat_v.shape[0]

    # mat_v : (Ns, Nv)
    # theta : (Nv,)
    res = mat_v @ theta  # -> (Ns,)

    # eta : (Nh,)
    # mat_v @ w : (Ns, Nh)
    temp = eta + mat_v @ w  # -> (Ns, Nh)

    res += torch.sum(torch.log1p(torch.exp(temp)), dim=1)  # -> (Ns,)

    res = torch.sum(res) / Ns

    return -log_Z - res


def moyenne_empirique(mat_v):
    """ 
    Calcule l'estimation emprique de v ~ RBM

    - params: 
        mat_v : (Ns, Nvec)
        Ns : number of samples 
    """ 
    return torch.mean(mat_v, dim=0)


def moyenne_empirique_fonction(mat_v, mat_h):
    """ 
    Calcule l'estimation emprique de (v, h) ~ RBM 

    - params:  
        mat_v : (Ns, Nvec_v)
        mat_h : (Ns, Nvec_h)
    """
    return (mat_v.T @ mat_h) / mat_v.shape[0]



def calcul_gradients(w, mat_v, inputs_data, eta):
    """ 
    Calcule les gradients des parametres w, eta, theta 
    
    - params: 
        inputs_data: matrice de données (visibles) : (Ns, Nf)
        mat_h_sachant_data:  (Ns, Nh)

    - retour : 
        3 vecteurs de gradients

    univariee -- > moy empirique (moyenne sur toutes les colonnes axis 0)
    bivariée  -- > moy empirique d'une fonction  
    """ 
    proba_h_sachant_data = sigmoid(eta + inputs_data @ w) 
    proba_h_sachant_v    = sigmoid(eta + mat_v @ w) 

    grad_w     = moyenne_empirique_fonction(inputs_data, proba_h_sachant_data) - moyenne_empirique_fonction(mat_v, proba_h_sachant_v)  # <viha>_D - <viha>_RBM
    grad_theta = moyenne_empirique(inputs_data)          - moyenne_empirique(mat_v)              # <vi>_D - <vi>_RBM 
    grad_eta   = moyenne_empirique(proba_h_sachant_data) - moyenne_empirique(proba_h_sachant_v)  # <ha>_D - <ha>_RBM

    return grad_w, grad_eta, grad_theta


def sample_ber(p):
    return (torch.rand_like(p) < p).to(dtype=p.dtype) # comparaison entre 2 trucs sur gpu --> resultat sur gpu 


def bgs(start_v, w, eta, theta, nstep=10):
    """ Applique l'algorithme Block Gibbs Sampling pour l'echantillonage. """
    mat_v = start_v 

    for _ in range(nstep):
        # calculer p(h|v)
        proba_h_sachant_v = sigmoid(eta + mat_v @ w)

        # sample p(h|v) 
        mat_h = sample_ber(proba_h_sachant_v)  # Taille Nh

        # calculer p(v|h)
        proba_v_sachant_h = sigmoid(theta + mat_h @ w.t())  

        # sample p(v|h)
        mat_v = sample_ber(proba_v_sachant_h)  # Taille Nv

    mat_h = sample_ber(sigmoid(eta + mat_v @ w))  # dernier pas sur h 

    return mat_h, mat_v


def descente_gradient_rbm(Nv, Nh, Ns, inputs_data, w0, eta0, theta0, mu, epochs):
    """
    Met à jour les gradients des parametres w, eta, theta du modèle RBM 
    - params : 
        mu : taux d'apprentissage 
        Ns : nombre d'échantillons 
    """
    w = w0
    eta = eta0
    theta = theta0
    llh = torch.zeros(epochs)
    combinaisons = torch.tensor(list(product([0, 1], repeat=(Nv + Nh))), dtype=torch.float32, device="cuda")  # combinaisons possibles de 0 et 1 dans un tuple de longueur 5 
    mat_v = torch.randint(0, 2, (Ns, Nv), dtype=torch.float32, device="cuda")

    for i in trange(epochs, desc="Training RBM"):
        # 1ere etape : echantillonage
        _, mat_v = bgs(mat_v, w, eta, theta, 10)  # ie. (h, v), a chaque fois nouveau

        # 2eme etape : calcul gradient 
        grad_w, grad_eta, grad_theta = calcul_gradients(w, mat_v, inputs_data, eta)

        # 3eme etape : MAJ du gradient
        w     = w     + mu * grad_w
        eta   = eta   + mu * grad_eta
        theta = theta + mu * grad_theta

        llh[i] = log_likelihood(mat_v, w, theta, eta, combinaisons)

    return w, eta, theta, llh


if __name__ == "__main__":
    """
    Brouillon

    - params: 
        Nv - nb de features
        Nh - nb de noeuds caches

        w - matrice de poids visible -> poids cache (Nv x Nh)
        theta - vecteur de taille Nv
        eta - vecteur de taille Nh

        v : vec de tt les visibles  (taille Nv)
        h : vec de tt les caches    (taille Nh)
    """
    # Jeu de données Bars and Stripes: 
    inputs = torch.tensor(generate_bars_and_stripes_complete(4), dtype=torch.float32, device="cuda")

    # Initialisation
    Nv = 16        # noeuds visibles == nombre de pixels (pour images 4x4)
    Nh = 7         # noeuds caches 
    mu = 0.01      # learning rate 
    w = torch.zeros((Nv, Nh), dtype=torch.float32, device="cuda") # matrice de poids 
    eta = torch.zeros(Nh, dtype=torch.float32, device="cuda")      
    theta = torch.log(moyenne_empirique(inputs)) - torch.log(1 - moyenne_empirique(inputs)) # formule d'init. optimale

    # Dimensions des inputs: (N, D) 
    nb_samples = inputs.shape[0]  # N ---> samples
    nb_pixels  = inputs.shape[1]  # D ---> features

    # Calcul des biais (eta & theta) et de la matrice des poids (w) 
    w, eta, theta, llh = descente_gradient_rbm(Nv=Nv, Nh=Nh, Ns=nb_samples, inputs_data=inputs, w0=w, eta0=eta, theta0=theta, mu=mu, epochs=10000)

    # ".cpu" pour ramener le tenseur sur le CPU
    #  afin d'utiliser numpy pour compatibilité h5py
    with h5py.File("rbm_parameters.h5", "w") as f:
        f["weight_matrix"]   = w.cpu().numpy()  
        f["eta_vector"]      = eta.cpu().numpy() 
        f["theta_vector"]    = theta.cpu().numpy() 
        f["log_likelihoods"] = llh.cpu().numpy() 