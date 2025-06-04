import numpy as np
import pennylane as qml
from itertools import product


def sigmoid(x):
    """ Calcule et evalue la fonction sigmoid au point x. """ 
    return 1 / (1 + np.exp(-x))


def Z(v, h, w, theta, eta):
    """ Calcule le denominateur de l'expression de la log-vraisemblance
        la somme sur toutes les possibilités de v,h (dans notre cas {0,1})
        renvoie un vecteur de taille Nv + Nh  """
    Nv = len(v)
    Nh = len(h)
    
    combinaisons = list(product([0, 1], repeat=(Nv + Nh))) # combinaisons possibles de 0 et 1 dans un tuple de longueur 5 
    res = 0 # somme de ttes les config possibles 

    for vec in combinaisons:
        v = vec[0  : Nv] # les Nv premiers elt. 
        h = vec[Nv :   ] # le reste des elt., donc les Nh derniers elements 

        res += H(v, h, w, theta, eta) # H prend en entrée une configuration possible

    return res



def log_likelihood(v, h, w, theta, eta):
    """ Calcule la log-vraisemblance selon la fonction de masse d'une RBM.
        renvoie un vecteur de taille Nh + Nv """
    return 1/Z(v, h) * np.exp(H(v, h, w, theta, eta))


def H(v, h, w, theta, eta):
    """ Fonction d'energie du RBM """
    res = 0
    for i in range(Nv):
        for a in range(Nh):
            res += v[i] * w[i][a] * h[a]

    return -(res + v@theta + h@eta)


def sample_ber(p : float):
    x = np.random.rand()
    if x < p:
        return 1 
    else:
        return 0


def bgs(w, eta, theta, nstep=10):
    """ Applique l'algorithme Block Gibbs Sampling pour l'echantillonage. """
    v = np.random.randint(0, 1, size=Nv) # initialisation 
    h = np.random.randint(0, 1, size=Nh)

    for n in range(nstep):
        # calculer p(h|v)
        proba_h_sachant_v = sigmoid(eta + v@w)

        # sample p(h|v) 
        couche_h = np.ones(Nh, dtype=int)
        for i in range(Nh):
            couche_h[i] = sample_ber(proba_h_sachant_v[i])

        # calculer p(v|h)
        proba_v_sachant_h = sigmoid(theta + w@h)  

        # sample p(v|h)
        couche_v = np.ones(Nv, dtype=int)
        for i in range(Nv):
            couche_v[i] = sample_ber(proba_v_sachant_h[i])

    return couche_h, couche_v


def moyenne_empirique(mat_v, Ns):
    """ 
    Calcule l'estimation emprique de v ~ RBM

    - params: 
        mat_v : (Ns, Nvec)
    """
    Nvec = mat_v.shape[1] # nb de composante 
    X = np.zeros(Nvec)

    for i in range(Nvec): # pour chaque composante 
        sum_vi = 0 

        for k in range(Ns): # on passe sur chacun de ses echantillons  
            sum_vi += mat_v[k][i]

        X[i] = sum_vi # Le i-eme elt. de X est la somme des v_i 

    return 1/Ns * X # moyenner toutes les composantes 


def moyenne_empirique_fonction(mat_v, mat_h, Ns):
    """ 
    Calcule l'estimation emprique de (v, h) ~ RBM 

    - params: 
        mat_v : (Ns, Nvec_v)
        mat_h : (Ns, Nvec_h)
    """
    Nvec_v = mat_v.shape[1] # nb de composante de v
    Nvec_h = mat_h.shape[1] # nb de composante de h

    X = np.zeros((Nvec_v, Nvec_h)) 
    matrices_des_k = np.zeros((Ns, Nvec_v, Nvec_h))

    for k in range(Ns): # pour chaque echantillon

        matrice_k = np.zeros((Nvec_v, Nvec_h)) # on a une matrice temporaire k

        for i in range(Nvec_v):
            for a in range(Nvec_h):
                matrice_k[i][a] = mat_v[k][i] * mat_h[k][a] # ses composantes = v_i^(k) * h_a^(k)

        matrices_des_k[k] = matrice_k # on va stocker ttes ces matrices k, dans une grosse matrice 

    for i in range(Nvec_v):
        for a in range(Nvec_h):
            sum_k = 0
            for k in range(Ns):
                # on parcourt toutes les k matrices stockees pour calculer la somme de leurs 
                # composantes, autrement dit on somme sur les k : v_i^(k) * h_a^(k) pour tout (i, a)
                sum_k += matrices_des_k[k][i][a]

            X[i][a] = sum_k
    
    return 1/Ns * X # moyenner le tout  
    

def calcul_gradients(w, mat_v, mat_h, Ns, inputs_data):
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
    proba_h_sachant_data = sigmoid(eta + inputs_data@w) 

    grad_w     = moyenne_empirique_fonction(inputs_data, proba_h_sachant_data, Ns) - moyenne_empirique_fonction(mat_v, mat_h, Ns)  # <viha>_D - <viha>_RBM
    grad_theta = moyenne_empirique(inputs_data, Ns)          - moyenne_empirique(mat_v, Ns)  # <vi>_D - <vi>_RBM 
    grad_eta   = moyenne_empirique(proba_h_sachant_data, Ns) - moyenne_empirique(mat_h, Ns)  # <ha>_D - <ha>_RBM

    return grad_w, grad_eta, grad_theta


def descente_gradient_rbm(h, v, inputs_data, Ns, w0, eta0, theta0, mu, nstep=10):
    """
    Met à jour les gradients des parametres w, eta, theta du modèle RBM 
    - params : 
        mu : taux d'apprentissage 
        Ns : nombre d'échantillons 
    les valeurs initiales des parametres sont définies avant.
    """
    mat_h = np.zeros((Ns, len(h)))
    mat_v = np.zeros((Ns, len(v)))
    
    w = w0
    eta = eta0
    theta = theta0

    for i in range(nstep):
        # 1ere etape : echantillonage
        for k in range(Ns):
            mat_h[k], mat_v[k] = bgs(w, eta, theta, nstep) # ie. (couche_h, couche_v), a chaque fois nouveau

        # 2eme etape : calcul gradient 
        grad_w, grad_eta, grad_theta = calcul_gradients(w, mat_v, mat_h, Ns, inputs_data)

        # 3eme etape : MAJ du gradient
        w = w + mu * grad_w
        eta = eta + mu * grad_eta
        theta = theta + mu * grad_theta

        log_llh = log_likelihood(v, h, w, theta, eta)
        print(log_llh)

        
    return w, eta, theta

    
     
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
    # Initialisation
    Nv = 16 # nombre de pixels (pour images 4x4)
    Nh = 3
    w = np.zeros((Nv, Nh))
    eta = np.zeros(Nh)
    theta = np.zeros(Nv)
    mu = 0.025 # ???????????
    
    # Couches des noeuds cachés (h) et visibles (v)
    couche_h, couche_v = bgs(w=w, eta=eta, theta=theta)  # ~ RBM

    print(f"Couche h : {couche_h}")
    print(f"Couche v : {couche_v}")

    # Jeu de données Bars and Stripes: 
    [dataset] = qml.data.load("other", name="bars-and-stripes") 
    inputs = dataset.train['4']['inputs'] # images de pixels 4x4 
    
    # Dimensions des inputs: (N, D) 
    inputs_np = np.array(inputs)
    nb_samples = inputs_np.shape[0] # N ---> samples
    nb_pixels  = inputs_np.shape[1] # D ---> features

    # Calcul des biais (eta & theta) et de la matrice des poids (w) 
    w, eta, theta = descente_gradient_rbm(h=couche_h, v=couche_v, Ns=nb_samples, inputs_data=inputs_np, w0=w, eta0=eta, theta0=theta, mu=mu, nstep=10)

    print(w)
    print(eta)
    print(theta)
