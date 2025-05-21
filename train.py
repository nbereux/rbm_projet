import numpy as np
import pennylane as qml

# dataset Bars and Stripes
dataset = qml.data.load("bars-and-stripes", n=4)[0]





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
    (couche_h, couche_v) = bgs(w=w, eta=eta, theta=theta)  #  ~ RBM

    print(f"Couche h : {couche_h}")
    print(f"Couche v : {couche_v}")

    # 2. Calcul gradient
    esp_h = np.mean(couche_h)
    esp_v = np.mean(couche_v)

    print(esp_h)
    print(esp_v)

    # dataset 
    images = dataset.data
    labels = dataset.labels











