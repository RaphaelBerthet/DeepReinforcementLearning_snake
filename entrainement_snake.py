import numpy as np
import random
import copy
import os
from collections import deque

NB_PARTIES = 200
NB_SAMPLES_MAX = 50000
p_debut = 0.01
p_fin = 0.01
NB_SAMPLES_DEBUT_ENTRAINEMENT = 5000
TAILLE_BATCHS = 128
NB_ENTRAINEMENT_BATCH = 4
gamma = 0.95
learning_rate = 0.0001
ACTU_W_TARGET = 1000  # tous les ... batchs
PERIODE_STOCKAGE_PC = 4800000 // ACTU_W_TARGET  # toutes les ... maj du reseau

'''
state (29): direction de la pomme (gauche droite haut bas) (booleen 0/1), 
danger immediat sur les cases distantes de 2 coups max (gauche, droite, haut, bas, gh, dh, gb, db, gg, dd, hh, bb)
booleen, distance au mur (gauche, droite ...), distance au corps normalisée (gauche, droite ...), longueur du serpent, 
surface_accessiblegauche/100, surface_accessibledroite/100, surface_accessiblehaut/100, surface_accessiblebas/100
actions (4):
1 : gauche
2 : droite
3 : haut
4 : bas
rewards:
cf
=> 128 neurones intermédiaires
=> 2e couche intermédiaire 64 neurones
sample = [state1, state2, action, reward, terminal_state] (61)
'''

print(os.getcwd())
os.chdir("3_snake")
print(os.getcwd())
try:
    data = np.load('reseau_neurones.npz')
    W1 = data['W1']
    W2 = data['W2']
    W3 = data['W3']
    B1 = data['B1']
    B2 = data['B2']
    B3 = data['B3']
except FileNotFoundError:
    W1 = np.random.randn(128, 29) * np.sqrt(2 / 29)  # He init pour ReLU
    W2 = np.random.randn(64, 128) * np.sqrt(2 / 128)
    W3 = np.random.randn(4, 64) * np.sqrt(2 / 64)
    B1 = np.zeros(128)
    B2 = np.zeros(64)
    B3 = np.zeros(4)

samples = np.zeros((NB_SAMPLES_MAX, 61), dtype=np.float32)  # 300+300+3
samples_count = 0  # Nombre réel de samples stockés
head = 0  # Index circulaire (tête)


def surface_accessible(serpent, pomme=None):
    """
    Retourne le nombre de cases accessibles depuis la tête du serpent.

    Le BFS tient compte :
    - de la libération progressive de la queue ;
    - de la pomme : si un chemin passe par la pomme, le serpent
      s'allonge d'un segment à cet instant. La libération des segments
      qui n'étaient pas encore libérés est alors retardée d'un coup.

    Un état du BFS est :
        (position, pomme_mangee)

    afin de distinguer les chemins qui ont mangé la pomme de ceux
    qui ne l'ont pas mangée.
    """

    corps = serpent.cases_occupe
    n = len(corps)

    # Temps auquel chaque segment du corps est normalement libéré.
    temps_liberation = {}

    for i, case in enumerate(corps[1:], start=1):
        temps_liberation[tuple(case)] = n - i

    tete = tuple(corps[0])
    pomme_pos = (pomme.x, pomme.y) if pomme is not None else None

    # État = (position, pomme_mangee)
    debut = (tete, tete == pomme_pos)

    # (position, pomme_mangee, temps)
    file = deque([(tete, tete == pomme_pos, 0)])

    visites = {debut}

    while file:
        position, pomme_mangee, temps = file.popleft()

        x, y = position

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:

            voisin = (x + dx, y + dy)

            # Limites de la grille
            if not (1 <= voisin[0] <= 10 and 1 <= voisin[1] <= 10):
                continue

            nouveau_temps = temps + 1

            # Le chemin mange la pomme en arrivant dessus.
            nouvelle_pomme_mangee = (
                pomme_mangee or voisin == pomme_pos
            )

            nouvel_etat = (voisin, nouvelle_pomme_mangee)

            if nouvel_etat in visites:
                continue

            # Vérification du corps.
            if voisin in temps_liberation:

                liberation = temps_liberation[voisin]

                # Si la pomme a déjà été mangée avant d'arriver
                # sur cette case, la croissance du serpent retarde
                # sa libération d'un coup.
                if pomme_mangee:
                    liberation += 1

                if nouveau_temps < liberation:
                    continue

            visites.add(nouvel_etat)

            file.append(
                (voisin, nouvelle_pomme_mangee, nouveau_temps)
            )

    # Une case peut être accessible dans les deux états.
    # On veut compter les cases, pas les états.
    positions_accessibles = {
        position for position, _ in visites
    }

    return len(positions_accessibles)


def build_state(pomme, serpent):
    state = np.zeros(29, dtype=np.float32)
    tete = serpent.cases_occupe[0]
    state[0] = int(pomme.x < tete[0])
    state[1] = int(pomme.x > tete[0])
    state[2] = int(pomme.y < tete[1])
    state[3] = int(pomme.y > tete[1])
    state[4] = int([tete[0] - 1, tete[1]] in serpent.cases_occupe[1:-1] or tete[0] == 1)  # danger immédiat gauche
    state[5] = int([tete[0] + 1, tete[1]] in serpent.cases_occupe[1:-1] or tete[0] == 10)  # danger immédiat droite
    state[6] = int([tete[0], tete[1] - 1] in serpent.cases_occupe[1:-1] or tete[1] == 1)  # danger immédiat haut
    state[7] = int([tete[0], tete[1] + 1] in serpent.cases_occupe[1:-1] or tete[1] == 10)  # danger immédiat bas
    state[8] = int([tete[0] - 1, tete[1] - 1] in serpent.cases_occupe[1:-1] or (tete[0] == 1 and tete[1] == 1))  # gh
    state[9] = int([tete[0] + 1, tete[1] - 1] in serpent.cases_occupe[1:-1] or (tete[0] == 10 and tete[1] == 1))  # dh
    state[10] = int([tete[0] - 1, tete[1] + 1] in serpent.cases_occupe[1:-1] or (tete[0] == 1 and tete[1] == 10))  # gb
    state[11] = int([tete[0] + 1, tete[1] + 1] in serpent.cases_occupe[1:-1] or (tete[0] == 10 and tete[1] == 10))  # db
    state[12] = int([tete[0] - 2, tete[1]] in serpent.cases_occupe[1:-1] or tete[0] <= 2)  # gg
    state[13] = int([tete[0] + 2, tete[1]] in serpent.cases_occupe[1:-1] or tete[0] >= 9)  # dd
    state[14] = int([tete[0], tete[1] - 2] in serpent.cases_occupe[1:-1] or tete[1] <= 2)  # hh
    state[15] = int([tete[0], tete[1] + 2] in serpent.cases_occupe[1:-1] or tete[1] >= 9)  # bb
    state[16] = (tete[0] - 1) / 9  # distance au mur gauche
    state[17] = (10 - tete[0]) / 9
    state[18] = (tete[1] - 1) / 9
    state[19] = (10 - tete[1]) / 9
    distg = 9
    distd = 9
    disth = 9
    distb = 9
    for element in serpent.cases_occupe[1:]:
        if element[1] == tete[1] and tete[0] > element[0]:  # element à gauche
            distg = min(distg, tete[0] - element[0] - 1)
        if element[1] == tete[1] and tete[0] < element[0]:  # element à droite
            distd = min(distd, element[0] - tete[0] - 1)
        if element[0] == tete[0] and tete[1] > element[1]:  # element en haut
            disth = min(disth, tete[1] - element[1] - 1)
        if element[0] == tete[0] and tete[1] < element[1]:  # element en bas
            distb = min(distb, element[1] - tete[1] - 1)
    state[20] = distg / 9
    state[21] = distd / 9
    state[22] = disth / 9
    state[23] = distb / 9
    state[24] = len(serpent.cases_occupe) / 100
    if state[4]:  # danger immédiat gauche
        state[25] = 0
    else:
        serpent_copie = copy.deepcopy(serpent)
        serpent_copie.direction = 1
        serpent_copie.avancer(True)
        state[25] = surface_accessible(serpent_copie, pomme) / 100

    if state[5]:
        state[26] = 0
    else:
        serpent_copie = copy.deepcopy(serpent)
        serpent_copie.direction = 2
        serpent_copie.avancer(True)
        state[26] = surface_accessible(serpent_copie, pomme) / 100

    if state[6]:
        state[27] = 0
    else:
        serpent_copie = copy.deepcopy(serpent)
        serpent_copie.direction = 3
        serpent_copie.avancer(True)
        state[27] = surface_accessible(serpent_copie, pomme) / 100
    if state[7]:
        state[28] = 0
    else:
        serpent_copie = copy.deepcopy(serpent)
        serpent_copie.direction = 4
        serpent_copie.avancer(True)
        state[28] = surface_accessible(serpent_copie, pomme) / 100
    return state


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


class Serpent:
    def __init__(self):
        self.direction = 2  # 1 : gauche; 2 : droite; 3 : haut; 4 : bas
        self.cases_occupe = [[5, 5], [4, 5], [3, 5]]  # le 1er correspond à la tête le 2e...

    def avancer(self, simulation_en_cours=False):
        global pomme, partie_en_cours, reward
        tete = self.cases_occupe[0]
        droite = True
        gauche = True
        haut = True
        bas = True
        for case in self.cases_occupe[1:-1]:  # la tete peut aller a la place de la queue
            if case[0] + 1 == tete[0] and case[1] == tete[1]:
                gauche = False
            if case[0] - 1 == tete[0] and case[1] == tete[1]:
                droite = False
            if case[1] - 1 == tete[1] and case[0] == tete[0]:
                bas = False
            if case[1] + 1 == tete[1] and case[0] == tete[0]:
                haut = False
        cases = copy.deepcopy(self.cases_occupe)
        tete = cases[0]
        if self.direction == 1 and tete[0] > 1 and gauche:
            tete[0] -= 1
        elif self.direction == 2 and tete[0] < 10 and droite:
            tete[0] += 1
        elif self.direction == 3 and tete[1] > 1 and haut:
            tete[1] -= 1
        elif self.direction == 4 and tete[1] < 10 and bas:
            tete[1] += 1
        if tete == self.cases_occupe[0] and not simulation_en_cours:
            if not simulation_en_cours:    
                reward += -2
                partie_en_cours = False

        elif tete == [pomme.x, pomme.y]:
            # on avance le serpent avant de regénérer une pomme
            self.cases_occupe = [tete] + self.cases_occupe
            if not simulation_en_cours:    
                reward += 10  ## modifier aussi dist_avant/dist_apres
                pomme = Pomme(self.cases_occupe)

        else:
            self.cases_occupe = [tete] + self.cases_occupe[:-1]


class Pomme:
    def __init__(self, cases_occupe):
        global partie_en_cours, reward
        cases_libres = [[i, j] for i in range(1, 11) for j in range(1, 11)]
        for case in cases_occupe:
            try:
                cases_libres.remove(case)
            except:
                print('bug chelou case pas dans cases libres')
                print(serpent.cases_occupe)
                print(cases_libres)
                print('on continue')
        if cases_libres == []:
            partie_en_cours = False
            reward += 100
            cases_libres.append([-1, -1])
            print("grille complétée !!!!!!!!!!")
        x, y = random.choice(cases_libres) 
        self.x = x
        self.y = y


W1_target, W2_target, W3_target = W1.copy(), W2.copy(), W3.copy()
B1_target, B2_target, B3_target = B1.copy(), B2.copy(), B3.copy()
compteur_target = 0
for partie in range(NB_PARTIES):
    p = p_debut - (p_debut - p_fin) * partie / NB_PARTIES

    if partie % 100 == 0:
        print(f"progression : {partie * 100 / NB_PARTIES} %")

    ## initialisation de la partie
    serpent = Serpent()
    pomme = Pomme(serpent.cases_occupe)

    derniers_deplacements = []  ## à l'intérieur : [xtete, ytete, taille_serpent]
    partie_en_cours = True
    while partie_en_cours:
        state1 = build_state(pomme, serpent)

        ## choix d'une action
        if random.random() <= p:
            tete = serpent.cases_occupe[0]
            gauche = int([tete[0] - 1, tete[1]] in serpent.cases_occupe[1:-1] or tete[0] == 1)
            droite = int([tete[0] + 1, tete[1]] in serpent.cases_occupe[1:-1] or tete[0] == 10)
            haut = int([tete[0], tete[1] - 1] in serpent.cases_occupe[1:-1] or tete[1] == 1)
            bas = int([tete[0], tete[1] + 1] in serpent.cases_occupe[1:-1] or tete[1] == 10)
            L = []
            if not gauche:
                L.append(1)
            if not droite:
                L.append(2)
            if not haut:
                L.append(3)
            if not bas:
                L.append(4)
            if len(L) == 0:
                L.append(random.randint(1, 4))
            action = random.choice(L)
        else:
            A0 = np.array(state1)
            Z1 = np.dot(W1, A0) + B1
            A1 = relu(Z1)
            Z2 = np.dot(W2, A1) + B2
            A2 = relu(Z2)
            Z3 = np.dot(W3, A2) + B3
            A3 = Z3
            action = np.argmax(A3) + 1
        serpent.direction = action

        ## on avance
        reward = 0
        tete = serpent.cases_occupe[0]
        dist_avant = abs(pomme.x - tete[0]) + abs(pomme.y - tete[1])
        surface_accessible_avant = surface_accessible(serpent, pomme)
        serpent.avancer()
        derniers_deplacements.append(
            [serpent.cases_occupe[0][0], serpent.cases_occupe[0][1], len(serpent.cases_occupe)])
        taille = len(derniers_deplacements)
        cycle_detecte = False
        for i in range(1, min(taille // 2, 100)):
            if derniers_deplacements[taille - i:] == derniers_deplacements[taille - 2 * i:taille - i]:
                cycle_detecte = True
        if cycle_detecte:
            reward -= 3  # motif qui se repète
        tete = serpent.cases_occupe[0]
        dist_apres = abs(pomme.x - tete[0]) + abs(pomme.y - tete[1])
        if partie_en_cours:
            reward += (surface_accessible(serpent, pomme) - surface_accessible_avant) / 10
        
        '''if reward < 10 and dist_avant > dist_apres:
                reward += 0.1
            else:
                reward -= 0.1'''
    
        tete = serpent.cases_occupe[0]
        terminal_state = not partie_en_cours
        state2 = build_state(pomme, serpent)
        sample = np.concatenate([
            state1,  # (61,)
            state2,  # (61,)
            [action, reward, terminal_state]
        ])
        samples[head] = sample
        head = (head + 1) % len(samples)
        if samples_count < len(samples):
            samples_count += 1

        if samples_count >= NB_SAMPLES_DEBUT_ENTRAINEMENT:
            for _ in range(NB_ENTRAINEMENT_BATCH):
                indices = np.random.choice(samples_count, TAILLE_BATCHS, replace=False)
                selection = samples[indices]  # (100, 603)

                # --- 1. Extraction des données du batch (vectorisé) ---
                states1_batch = np.array([sample[:29] for sample in selection])  # (100, 306)
                states2_batch = np.array([sample[29:58] for sample in selection])  # (100, 306)
                actions_batch = np.array([int(sample[58]) for sample in selection])  # (100,)
                rewards_batch = np.array([sample[59] for sample in selection])  # (100,)
                terminal_states_batch = np.array([bool(sample[60]) for sample in selection])  # (100,)

                # --- 2. Forward Pass pour Q_target (réseau cible) ---
                Q_target = np.zeros(TAILLE_BATCHS)
                non_terminal_mask = ~terminal_states_batch
                Z1 = states2_batch @ W1.T + B1  # (100, 200)
                A1 = relu(Z1)
                Z2 = A1 @ W2.T + B2  # (100, 4)
                A2 = relu(Z2)
                Z3 = A2 @ W3.T + B3
                if np.any(non_terminal_mask):
                    Z1_target = states2_batch @ W1_target.T + B1_target  # (100, 200)
                    A1_target = relu(Z1_target)
                    Z2_target = A1_target @ W2_target.T + B2_target  # (100, 4)
                    A2_target = relu(Z2_target)
                    Z3_target = A2_target @ W3_target.T + B3_target
                    best_actions = np.argmax(Z3, axis=1)
                    Q_target[non_terminal_mask] = (rewards_batch[non_terminal_mask]
                        + gamma * Z3_target[non_terminal_mask, best_actions[non_terminal_mask]])
                Q_target[terminal_states_batch] = rewards_batch[terminal_states_batch]

                # --- 3. Forward Pass pour Q (réseau principal) ---
                Z1 = states1_batch @ W1.T + B1  # (100, 200)
                A1 = relu(Z1)
                Z2 = A1 @ W2.T + B2  # (100, 4)
                A2 = relu(Z2)
                Z3 = A2 @ W3.T + B3
                Q = Z3[np.arange(TAILLE_BATCHS), actions_batch - 1]  # Extraction des Q pour chaque action

                # --- 4. Backward Pass (vectorisé) ---
                gradientaC = Q - Q_target  # (100,)

                # delta2 : (100, 4)
                delta3 = np.zeros((TAILLE_BATCHS, 4))
                delta3[np.arange(TAILLE_BATCHS), actions_batch - 1] = gradientaC

                # delta1 : (100, 200)
                delta2 = (delta3 @ W3) * relu_derivative(Z2)
                delta1 = (delta2 @ W2) * relu_derivative(Z1)

                # Gradients pour W1, W2, B1, B2
                dW1 = delta1.T @ states1_batch  # (200, 300)
                dW2 = delta2.T @ A1  # (4, 200)
                dW3 = delta3.T @ A2
                dB1 = np.sum(delta1, axis=0)  # (200,)
                dB2 = np.sum(delta2, axis=0)  # (4,)
                dB3 = np.sum(delta3, axis=0)

                # --- 5. Mise à jour des poids ---
                W1 -= learning_rate * dW1 / TAILLE_BATCHS
                W2 -= learning_rate * dW2 / TAILLE_BATCHS
                W3 -= learning_rate * dW3 / TAILLE_BATCHS
                B1 -= learning_rate * dB1 / TAILLE_BATCHS
                B2 -= learning_rate * dB2 / TAILLE_BATCHS
                B3 -= learning_rate * dB3 / TAILLE_BATCHS

                # Mise à jour du réseau cible
                compteur_target += 1
                if compteur_target % ACTU_W_TARGET == 0:
                    W1_target, W2_target, W3_target = W1.copy(), W2.copy(), W3.copy()
                    B1_target, B2_target, B3_target = B1.copy(), B2.copy(), B3.copy()

                if compteur_target % PERIODE_STOCKAGE_PC == 0:
                    np.savez("reseau_neurones.npz", W1=W1, W2=W2, W3=W3, B1=B1, B2=B2, B3=B3)
                    print(f"partie : {partie}   Poids, biais exportés dans reseau_neurones.npz")

np.savez("reseau_neurones.npz", W1=W1, W2=W2, W3=W3, B1=B1, B2=B2, B3=B3)
print("Poids, biais exportés dans reseau_neurones.npz")