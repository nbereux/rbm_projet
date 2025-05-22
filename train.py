import numpy as np
import pennylane as qml


def sigmoid(x : float):
    return 1 / (1 + np.exp(-x))

def H(v, h, w, theta, eta):
    """ Fonction d'energie du RBM """
    res = 0
    for i in range(Nv):
        for a in range(Nh):
            res += v[i] * w[i][a] * h[a]

    return -(res + v@theta + h@eta)

def p(v, h):
    pass


def sample_ber(p : float):
    x = np.random.rand()
    if x < p:
        return 1 
    else:
        return 0

def bgs(w, eta, theta, nstep=10):
    """ Applique l'algorithme Block Gibbs Sampling pour l'echantillonage. """
    v = np.random.randint(0, 1, size=Nv)
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


def estim_emp(v, h):
    """ Calcule l'estimation emprique de (v, h) ~ P rbm """ 

    # v_Nv ^ (1) pour 1 sample
    # v_Nv ^ (15)
    # v.shape = (15, Nv)

    # h_Nh ^ (1) pour 1 sample
    # h_Nh ^ (15)
    # h.shape = (15, Nh)

    # c'est juste Nv qui nest pas forcement = a Nh
    # mais nb sample = entre eux 

    Nv = len(v)
    Nh = len(h)
    Ns = 1 # nb samples 
    res = np.zeros((Nv, Nh))

    X_res = np.zeros(Ns)

    #   X_res += X^(k) 
    for k in range(Ns):

        for i in range(Nv):
            for a in range(Nh):
                res[i][a] = v[i] * h[a]

        X_res[k] += res 


    # 1/Ns sum_k X^(k)

    return 1/Ns * res 



def estim_emp_vecteur(mat_v, Ns):
    """ 
    Calcule l'estimation emprique de vec ~ P rbm 
    mat_v : (Nvec, Ns)                                             (changer pour que ce soit Ns, Nvec)
    """
    Nvec = mat_v.shape[0] # nb de composante 
    X = np.zeros(Nvec)

    for i in range(Nvec): # pour chaque composante 

        sum_vi = 0 # la somme des v_i 

        for k in range(Ns): # on passe sur chaque echantillon pour cette composante 
            sum_vi += mat_v[i][k]

        X[i] = sum_vi 

    return 1/Ns * X # moyenner toutes les composantes 



def estim_emp_v_h(mat_v, mat_h, Ns):
    """ 
    Calcule l'estimation emprique de (v, h) ~ P rbm 
    mat_v : (Ns, Nvec_v)
    mat_h : (Ns, Nvec_h)
    """
    Nvec_v = mat_v.shape[1] # nb de composante  de v
    Nvec_h = mat_h.shape[1] # nb de composante  de h

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


def esperance_data(array_inputs):
    """ Calcule l'espérance du jeu de données.  
        parametres : tableau des inputs du dataset"""
    N = inputs_np.shape[0]  
    D = inputs_np.shape[1]
    res = 0 
    for i in range(N):
        for j in range(D):
            res += array_inputs[i][j]

    return 1/(N*D) * res 

def gradient_param(v, h):
    """ Calcule les gradients des parametres w, eta, theta """
    grad_w  = estim_emp(v, h)
    grad_theta = np.mean(v) - np.mean()
    grad_eta = np.mean(v) - np.mean()
    


if __name__ == "__main__":
    """
    Brouillon
    Partie 1 
    - params: 
        Nv - nb de features
        Nh - nb de noeuds caches

        w - matrice de poids visible -> poids cache (Nv x Nh)
        theta - vecteur de taille Nv
        eta - vecteur de taille Nh

        v : vec de tt les visibles  (taille Nv)
        h : vec de tt les caches    (taille Nh)

    Partie 2

    """
    Nv = 5
    Nh = 3

    w = np.zeros((Nv, Nh))
    eta = np.zeros(Nh)
    theta = np.zeros(Nv)
    
    # 1. Sample nodes
    (couche_h, couche_v) = bgs(w=w, eta=eta, theta=theta)  # ~ RBM

    print(f"Couche h : {couche_h}")
    print(f"Couche v : {couche_v}")

    # 2. Calcul de la partie négative du gradient (estimation empirique)
    esperance_rbm  = estim_emp(couche_v, couche_h)
    print(f"Estimation empirique: {esperance_rbm}" )

    # 3. Calcul de la partie positive du gradient (Esperance de la proba du dataset)
    [dataset] = qml.data.load("other", name="bars-and-stripes") # dataset Bars and Stripes

    inputs = dataset.train['4']['inputs'] # vector representations of 4x4 pixel images
    labels = dataset.train['4']['labels'] # labels for the above images
    
    # Dimensions de inputs: (N, D) 
    inputs_np = np.array(inputs)
    print(inputs_np.shape) 
    nb_samples = inputs_np.shape[0] # N
    nb_pixels = inputs_np.shape[1]  # D

    esperance_data = esperance_data(inputs_np)
    print(f"Esperance des valeurs du dataset Bars and Stripes : {esperance_data}")

    # 4. Calcul des gradients de w, eta, theta 
    (grad_w, grad_eta, grad_theta) = gradient_param()













