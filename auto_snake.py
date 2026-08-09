import random
import pyxel
import time
import copy
import numpy as np
from collections import deque

pyxel.init(121, 121, title='snake')

PERIODE_ACTU = 0.01  # bouge le serpent toutes les ...

data = np.load("reseau_neurones.npz")
W1 = data['W1']
W2 = data['W2']
W3 = data['W3']
B1 = data['B1']
B2 = data['B2']
B3 = data['B3']

Tracer_orientation_serpent = {
    "gauche" : (0, 5, 6, 1),
    "droite" : (5, 5, 6, 1),
    "haut" : (5, 0, 1, 6),
    "bas" : (5, 5, 1, 6)
}

class Serpent:
    def __init__(self):
        self.direction = 2  # 1 : gauche; 2 : droite; 3 : haut; 4 : bas
        self.cases_occupe = [[5, 5], [4, 5], [3, 5]]  # le 1er correspond à la tête le 2e...

    def actu_direction(self):
        state1 = build_state(pomme, serpent)
        A0 = np.array(state1)
        Z1 = np.dot(W1, A0) + B1
        A1 = relu(Z1)
        Z2 = np.dot(W2, A1) + B2
        A2 = relu(Z2)
        Z3 = np.dot(W3, A2) + B3
        A3 = Z3
        self.direction = np.argmax(A3) + 1

    def avancer(self, simulation_en_cours=False):
        global pomme
        tete = self.cases_occupe[0]
        droite = True
        gauche = True
        haut = True
        bas = True
        for case in self.cases_occupe[1:-1]:
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
            print('fin de partie')
            time.sleep(1)
            pyxel.quit()
        elif tete == [pomme.x, pomme.y]:
            self.cases_occupe = [tete] + self.cases_occupe
            if not simulation_en_cours:
                pomme = Pomme(self.cases_occupe)
        else:
            self.cases_occupe = [tete] + self.cases_occupe[:-1]

    def tracer(self):
        case = self.cases_occupe[0]
        case2 = self.cases_occupe[-1]
        pyxel.rect(conversion_numerocase_coordonnees(case[0]), conversion_numerocase_coordonnees(case[1]), 11, 11, 2)
        pyxel.rect(conversion_numerocase_coordonnees(case2[0]), conversion_numerocase_coordonnees(case2[1]), 11, 11, 1)
        for i in range(len(self.cases_occupe[1:-1])):
            indice_reel = i + 1
            x = conversion_numerocase_coordonnees(self.cases_occupe[indice_reel][0])
            y = conversion_numerocase_coordonnees(self.cases_occupe[indice_reel][1])
            pyxel.rect(x, y, 11, 11, 4)
            direction_case_avant = trouve_direction(self.cases_occupe[indice_reel], self.cases_occupe[indice_reel - 1])
            direction_case_apres = trouve_direction(self.cases_occupe[indice_reel], self.cases_occupe[indice_reel + 1])
            for direction in [direction_case_avant, direction_case_apres]:
                values = Tracer_orientation_serpent[direction]
                dx, dy, L, l = values[0], values[1], values[2], values[3]
                pyxel.rect(x + dx, y + dy, L, l, 3)

class Pomme:
    def __init__(self, cases_occupe):
        cases_libres = [[i, j] for i in range(1, 11) for j in range(1, 11)]
        for case in cases_occupe:
            cases_libres.remove(case)
        if cases_libres == []:
            print('Gagnez !!! Loser Loser hehe')
            time.sleep(2)
            pyxel.quit()
        x, y = random.choice(cases_libres)
        self.x = x
        self.y = y

    def tracer(self):
        pyxel.circ(conversion_numerocase_coordonnees(self.x) + 5, conversion_numerocase_coordonnees(self.y) + 5, 3, 8)

def surface_accessible(serpent):
    """
    Retourne le nombre de cases accessibles depuis la tête du serpent.

    Le BFS tient compte de la libération progressive de la queue :
    un segment du corps devient franchissable lorsqu'il est censé
    avoir quitté sa case.
    """

    corps = serpent.cases_occupe
    n = len(corps)

    # Temps de libération de chaque segment du corps (hors tête)
    temps_liberation = {}
    for i, case in enumerate(corps[1:], start=1):
        # queue -> 1
        # avant-queue -> 2
        # ...
        # segment derrière la tête -> n-1
        temps_liberation[tuple(case)] = n - i

    tete = tuple(corps[0])

    visites = {tete: 0}
    file = deque([tete])

    while file:
        x, y = file.popleft()
        profondeur = visites[(x, y)]

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            voisin = (x + dx, y + dy)

            if not (1 <= voisin[0] <= 10 and 1 <= voisin[1] <= 10):
                continue

            if voisin in visites:
                continue

            # Arrivée sur cette case après un déplacement supplémentaire
            temps_arrivee = profondeur + 1

            if (voisin in temps_liberation and
                    temps_arrivee < temps_liberation[voisin]):
                continue

            visites[voisin] = temps_arrivee
            file.append(voisin)

    return len(visites)

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
        state[25] = surface_accessible(serpent_copie) / 100

    if state[5]:
        state[26] = 0
    else:
        serpent_copie = copy.deepcopy(serpent)
        serpent_copie.direction = 2
        serpent_copie.avancer(True)
        state[26] = surface_accessible(serpent_copie) / 100

    if state[6]:
        state[27] = 0
    else:
        serpent_copie = copy.deepcopy(serpent)
        serpent_copie.direction = 3
        serpent_copie.avancer(True)
        state[27] = surface_accessible(serpent_copie) / 100
    if state[7]:
        state[28] = 0
    else:
        serpent_copie = copy.deepcopy(serpent)
        serpent_copie.direction = 4
        serpent_copie.avancer(True)
        state[28] = surface_accessible(serpent_copie) / 100
    return state

def trouve_direction(origine, point):
    '''renvoie la direction de point par rapport à origine (les 2 sont collés)'''
    if [a + b for a, b in zip(origine, [1, 0])] == point:
        return "droite"
    elif [a + b for a, b in zip(origine, [-1, 0])] == point:
        return "gauche"
    elif [a + b for a, b in zip(origine, [0, 1])] == point:
        return "bas"
    elif [a + b for a, b in zip(origine, [0, -1])] == point:
        return "haut"

def relu(z):
    return np.maximum(0, z)


def conversion_numerocase_coordonnees(x):
    return 1 + 12 * (x - 1)


def draw():
    pyxel.cls(0)
    ## grille
    pyxel.rect(0, 0, 121, 1, 11)
    pyxel.rect(0, 0, 1, 121, 11)
    pyxel.rect(120, 0, 1, 121, 11)
    pyxel.rect(0, 120, 121, 1, 11)
    for i in range(1, 10):
        pyxel.rect(i * 12, 1, 1, 119, 11)
        pyxel.rect(1, i * 12, 119, 1, 11)

    serpent.tracer()
    pomme.tracer()


def update():
    global last_t
    if pyxel.btn(pyxel.KEY_Q):
        pyxel.quit()

    serpent.actu_direction()
    if time.perf_counter() - PERIODE_ACTU > last_t:
        serpent.avancer()
        last_t = time.perf_counter()


serpent = Serpent()
pomme = Pomme(serpent.cases_occupe)
last_t = time.perf_counter()

pyxel.run(update, draw)
