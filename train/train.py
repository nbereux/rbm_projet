import h5py 
import numpy as np
import pennylane as qml

from tqdm import trange
from itertools import product
from scipy.special import logsumexp, log1p


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
    return 1 / (1 + np.exp(-x))


def all_possible_config_negative(w, theta, eta, combinaisons):
    """ Calcule le log du denominateur de l'expression de la log-vraisemblance
        la somme sur toutes les possibilités de v,h (dans notre cas {0,1})
        renvoie un vecteur de taille Nv + Nh  """
    Nv, _ = w.shape
    
    # res = 0 # somme de ttes les config possibles 
    # for vec in combinaisons:
    #     v = vec[0  : Nv] # les Nv premiers elt. 
    #     h = vec[Nv :   ] # le reste des elt., donc les Nh derniers elements 

    #     res += np.exp(-H(v, h, w, theta, eta)) # H prend en entrée une configuration possible

    # return res

    parties_v = combinaisons[:, :Nv] # ttes les combinaisons en gardant que les Nv premiers elt. 
    parties_h = combinaisons[:, Nv:] # idem pour les Nh derniers elt. 

    # H prend en entrée une configuration possible
    # on somme donc sur ttes les configurations possibles

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
        # res = 0
    # for k in range(Ns):
        # for i in range(Nv):
        #     for a in range(Nh):
        #         res += v[k][i] * w[i][a] * h[k][a]
    # return -(res + v@theta + h@eta)

    return -(np.einsum("ki,ia,ka->k", mat_v, w, mat_h) + mat_v@theta + mat_h@eta) 



def log_likelihood(mat_v, w, theta, eta, combinaisons):
    """ Calcule la log-vraisemblance selon la fonction de masse d'une RBM.
        renvoie un vecteur de taille Nh + Nv """
    
     
    #   log( 1/Z(w, theta, eta) * np.exp(-H(v, h, w, theta, eta)) )
    # = log( 1/Z(w, theta, eta) ) + log (np.exp(-H(v, h, w, theta, eta)))
    # = -log( Z(w, theta, eta) ) - H(v, h, w, theta, eta)
    # = -log ( sum([np.exp(-H(v, h, w, theta, eta)) for all (v, h)] ) - H(v, h, w, theta, eta)

    log_Z = logsumexp(all_possible_config_negative(w, theta, eta, combinaisons))
    Ns = mat_v.shape[0]


    # mat_v : (Ns, Nv)
    # theta : (Nv,)
    res = mat_v@theta # -> (Ns,)

    
    # for a in range(Nh):
        # mat_v : (Ns, Nv)
        # w     : (Nv, Nh)

        # eta ; (Nh,)
        # mat_v @ w : (Ns, Nh)

    res3 = eta + mat_v @ w # -> (Ns, Nh)


    res += np.sum(log1p(np.exp(res3)), axis=1) # -> (Ns,)

    res = np.sum(res)

    return -log_Z - 1/Ns * res


def moyenne_empirique(mat_v):
    """ 
    Calcule l'estimation emprique de v ~ RBM

    - params: 
        mat_v : (Ns, Nvec)
        Ns : number of samples 
    """ 
    return np.mean(mat_v, axis=0)


def moyenne_empirique_fonction(mat_v, mat_h):
    """ 
    Calcule l'estimation emprique de (v, h) ~ RBM 

    - params:  
        mat_v : (Ns, Nvec_v)
        mat_h : (Ns, Nvec_h)
    """
    # Ns = mat_v.shape[0]
    # Nvec_v = mat_v.shape[1] # nb de composante de v
    # Nvec_h = mat_h.shape[1] # nb de composante de h

    # X = np.zeros((Nvec_v, Nvec_h)) 
    # matrices_des_k = np.zeros((Ns, Nvec_v, Nvec_h))

    # for k in range(Ns): # pour chaque echantillon

    #     matrice_k = np.zeros((Nvec_v, Nvec_h)) # on a une matrice temporaire k

    #     for i in range(Nvec_v):
    #         for a in range(Nvec_h):
    #             matrice_k[i][a] = mat_v[k][i] * mat_h[k][a] # ses composantes = v_i^(k) * h_a^(k)

    #     matrices_des_k[k] = matrice_k # on va stocker ttes ces matrices k, dans une grosse matrice 

    # de dimension (Ns, Nv, Nh)
    #######matrices_des_k = mat_v[:, :, None] * mat_h[:, None, :] # 'None', en slicing, cree un axe de longueur 1

    # for i in range(Nvec_v):
    #     for a in range(Nvec_h):
    #         sum_k = 0
    #         for k in range(Ns):
    #             # on parcourt toutes les k matrices stockees pour calculer la somme de leurs 
    #             # composantes, autrement dit on somme sur les k : v_i^(k) * h_a^(k) pour tout (i, a)
    #             sum_k += matrices_des_k[k][i][a]

    #         X[i][a] = sum_k
    
    # return 1/Ns * X # moyenner le tout  

    ####return np.mean(matrices_des_k, axis=0) # on moyenne sur les Ns samples 

    return (mat_v.T @ mat_h) / mat_v.shape[0]


    

def calcul_gradients(w, mat_v, inputs_data):
    """ 
    Calcule les gradients des parametres w, eta, theta, composés respectivement de phase positive et phase négative
    Note: la phase négative est approximée par l'échantillonnage de Block Gibbs Sampling

    
    - params: 
        inputs_data: matrice de données (visibles) : (Ns, Nf)
        mat_v:  (Ns, Nv) la matrice des échantillons purs RBM de la couche visibles 

    - retour : 
        3 vecteurs de gradients

    univariee -- > moy empirique (moyenne sur toutes les colonnes axis 0)
    bivariée  -- > moy empirique d'une fonction  
    """ 
    proba_h_sachant_data = sigmoid(eta + inputs_data@w) 
    proba_h_sachant_v    = sigmoid(eta + mat_v@w) 

    grad_w     = moyenne_empirique_fonction(inputs_data, proba_h_sachant_data) - moyenne_empirique_fonction(mat_v, proba_h_sachant_v)  # <viha>_D - <viha>_RBM
    grad_theta = moyenne_empirique(inputs_data)          - moyenne_empirique(mat_v)              # <vi>_D - <vi>_RBM 
    grad_eta   = moyenne_empirique(proba_h_sachant_data) - moyenne_empirique(proba_h_sachant_v)  # <ha>_D - <ha>_RBM

    return grad_w, grad_eta, grad_theta


def sample_ber(p):
    return (np.random.rand(*p.shape) < p).astype(int)


def bgs(start_v, w, eta, theta, Ns, Nv, Nh, nstep=10):
    """ 
    Applique l'algorithme Block Gibbs Sampling pour l'echantillonage du modèle RBM 
    afin d'approximer la phase négative du gradient.
        
    retour: des échantilllons du modèle RBM pur sans dépendances aux données.
            - couche V et couche H 

     """
    # mat_v = np.random.randint(0, 1, size=(Ns, Nv)) # initialisation 
    # mat_h = np.random.randint(0, 1, size=(Ns, Nh))
    mat_v = start_v 

    for _ in range(nstep):
        # calculer p(h|v)
        proba_h_sachant_v = sigmoid(eta + mat_v@w)

        # sample p(h|v) 
        mat_h = sample_ber(proba_h_sachant_v) # Taille Nh

        # calculer p(v|h)
        proba_v_sachant_h = sigmoid(theta + mat_h@w.T)  

        # sample p(v|h)
        mat_v = sample_ber(proba_v_sachant_h) # Taille Nv

    mat_h = sample_ber(sigmoid(eta + mat_v@w)) # dernier pas sur h 

    return mat_h, mat_v


def descente_gradient_rbm(Nv, Nh, Ns, inputs_data, w0, eta0, theta0, mu, epochs):
    """
    Méthode d'optimisation des paramètres du modèle en cherchant le maximum de la log_likelihood 
    Met à jour des parametres w, eta, theta du modèle RBM  en utilisatnt le calcul de gradient qui nous indique les directions d'optimisation
    - params : 
        mu : taux d'apprentissage 
        Ns : nombre d'échantillons 
    """
    mat_v = np.zeros((Ns, Nv))
    
    Ns = mat_v.shape[0]
    w = w0
    eta = eta0
    theta = theta0
    llh = np.zeros(epochs)
    combinaisons = np.array(list(product([0, 1], repeat=(Nv + Nh)))) # combinaisons possibles de 0 et 1 dans un tuple de longueur 5 

    mat_v = np.random.randint(0, 1, size=(Ns, Nv))

    for i in trange(epochs, desc="Training RBM"):
        # 1ere etape : echantillonage 
        _, mat_v = bgs(mat_v, w, eta, theta, Ns, Nv, Nh, 10) # ie. (h, v), a chaque fois nouveau, utilisés pour le calcul de gradient

        # 2eme etape : calcul gradient 
        grad_w, grad_eta, grad_theta = calcul_gradients(w, mat_v, inputs_data)

        # 3eme etape : Mise à Jour des paramètres du modèle 
        w = w + mu * grad_w
        eta = eta + mu * grad_eta
        theta = theta + mu * grad_theta

        llh[i] = log_likelihood(mat_v, w, theta, eta, combinaisons) #Les valeurs de la log_likelihood à chaque itération pour suivre son comportement.

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
    inputs = generate_bars_and_stripes_complete(4)
    
    
    # Initialisation
    Nv = 16                # noeuds visibles == nombre de pixels (pour images 4x4)
    Nh = 7                 # noeuds caches 
    mu = 0.01              # learning rate 
    w = np.zeros((Nv, Nh)) # matrice de poids 
    eta = np.zeros(Nh)     
    theta = np.log(moyenne_empirique(inputs)) - np.log(1 - moyenne_empirique(inputs))

    # Dimensions des inputs: (N, D) 
    inputs_np = np.array(inputs)
    nb_samples = inputs_np.shape[0] # N ---> samples
    nb_pixels  = inputs_np.shape[1] # D ---> features

    # Calcul des biais (eta & theta) et de la matrice des poids (w) 
    w, eta, theta, llh = descente_gradient_rbm(Nv=Nv, Nh=Nh, Ns=nb_samples, inputs_data=inputs_np, w0=w, eta0=eta, theta0=theta, mu=mu, epochs=100)

    with h5py.File("rbm_parameters.h5", "w") as f:
        f["weight_matrix"] = w  
        f["eta_vector"]    = eta 
        f["theta_vector"]  = theta 
        f["log_likelihoods"] = llh 
