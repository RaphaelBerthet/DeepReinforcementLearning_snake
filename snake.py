import random
import pyxel
import time
import copy

pyxel.init(111, 111, title='snake')

PERIODE_ACTU = 0.2  # bouge le serpent toutes les ...

Tracer_orientation_serpent = {
    "gauche" : (0, 5, 6, 1),
    "droite" : (5, 5, 6, 1),
    "haut" : (5, 0, 1, 6),
    "bas" : (5, 5, 1, 6)
}

class Serpent:
    def __init__(self):
        self.nx = 5
        self.ny = 5
        self.direction = 2  # 1 : gauche; 2 : droite; 3 : haut; 4 : bas
        self.cases_occupe = [[5, 5], [4, 5], [3, 5]]  # le 1er correspond à la tête le 2e...

    def tracer(self):
        case = self.cases_occupe[0]
        pyxel.rect(conversion_numerocase_coordonnees(case[0]), conversion_numerocase_coordonnees(case[1]), 10, 10, 2)
        for case in self.cases_occupe[1:]:
            pyxel.rect(conversion_numerocase_coordonnees(case[0]), conversion_numerocase_coordonnees(case[1]), 10, 10, 4)

    def actu_direction(self):
        tete = self.cases_occupe[0]
        if pyxel.btn(pyxel.KEY_LEFT):
            self.direction = 1
        elif pyxel.btn(pyxel.KEY_RIGHT):
            self.direction = 2
        elif pyxel.btn(pyxel.KEY_UP):
            self.direction = 3
        elif pyxel.btn(pyxel.KEY_DOWN):
            self.direction = 4

    def avancer(self):
        global pomme
        tete = self.cases_occupe[0]
        droite = True
        gauche = True
        haut = True
        bas = True
        for case in self.cases_occupe[1:]:
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
        if tete == self.cases_occupe[0]:
            print('fin de partie')
            time.sleep(1)
            pyxel.quit()
        elif tete == [pomme.x, pomme.y]:
            self.cases_occupe = [tete] + self.cases_occupe
            pomme = Pomme(self.cases_occupe)
        else:
            self.cases_occupe = [tete] + self.cases_occupe[:-1]

class Pomme:
    def __init__(self, cases_occupe):
        cases_libres = [[i, j] for i in range(1, 11) for j in range(1, 11)]
        for case in cases_occupe:
            cases_libres.remove(case)
        x, y = random.choice(cases_libres)
        self.x = x
        self.y = y

    def tracer(self):
        pyxel.circ(conversion_numerocase_coordonnees(self.x) + 5, conversion_numerocase_coordonnees(self.y) + 5, 3, 8)

def conversion_numerocase_coordonnees(x):
    return 1 + 11 * (x - 1)

def draw():
    pyxel.cls(0)
    ## grille
    pyxel.rect(0, 0, 111, 1, 11)
    pyxel.rect(0, 0, 1, 111, 11)
    pyxel.rect(110, 0, 1, 111, 11)
    pyxel.rect(0, 110, 111, 1, 11)
    for i in range(1, 10):
        pyxel.rect(i * 11, 1, 1, 109, 11)
        pyxel.rect(1, i * 11, 109, 1, 11)

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