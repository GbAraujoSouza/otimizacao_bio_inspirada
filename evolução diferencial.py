import numpy as np
import random as rd

# função a ser otimizada (minimizada) ------------------------------------------------------------------------
def func_objetivo(vetor: np.array) -> float: # vetor = [x,y]
    return (-20*np.exp(-0.2*np.sqrt(0.5*(vetor[0]**2 + vetor[0]**2))) - np.exp(0.5*(np.cos(2*np.pi*vetor[0])+np.cos(2*np.pi*vetor[0]))) + np.exp(1) + 20)

dimencao = 2 # dimenção D do problema / tamanho do vetor (indivíduo)

#intervalos do domínio ---------------------------------------------------------------------------------------
limInferior = -5
limSuperior = 5

# parâmetros do algorítmo ------------------------------------------------------------------------------------
tamPopulacao = 20# >= 4
probCrossover = 0.9 # [0,1]
ampF = 0.8 # fator de amplificação [0,2]
numGeracoes = 200

# Operadores genéticos ---------------------------------------------------------------------------------------
def mutacao(individuoA: np.array, individuoB: np.array, individuoC: np.array) -> np.array:
    return individuoA + ampF*(individuoB-individuoC)
    
def crossover(individuoOriginal: np.array, individuoMutante: np.array) -> np.array:
    novoIndividuo = np.zeros(dimencao)
    rind = rd.randint(0,dimencao-1)
    for posicao in range(dimencao):
        rpos = rd.uniform(0,1)
        if rpos <= probCrossover or posicao == rind:
            novoIndividuo[posicao] = individuoMutante[posicao]
        elif rpos > probCrossover and posicao != rind:
            novoIndividuo[posicao] = individuoOriginal[posicao]
    return novoIndividuo

# Iniciar população -----------------------------------------------------------------------------------------
pop = np.zeros((tamPopulacao, dimencao))
for individuo in range(tamPopulacao):
    for elemento in range(dimencao):
        pop[individuo,elemento] = rd.uniform(limInferior,limSuperior)
# print(pop,"\n")
rng = np.random.default_rng() # Gerador de números aleatórios

# Loop principal --------------------------------------------------------------------------------------------
for geracao in range(numGeracoes):
    # Avaliar critério de parada----------------------------------
    # Atualizar indivíduos ---------------------------------------
    proxPop = np.zeros((tamPopulacao,dimencao))
    for individuo in range(tamPopulacao):
        # Mutação - escolher indivíduos A, B, C ------------------
        array = rng.choice(pop,3,replace=False)
        while (pop[individuo] in array):
            array = rng.choice(pop,3,replace=False)
        mutante = mutacao(array[0],array[1],array[2])

        # Crossover ---------------------------------------------
        novoIndividuo = crossover(individuoOriginal=pop[individuo], individuoMutante=mutante)

        # Seleção -----------------------------------------------
        if func_objetivo(novoIndividuo) < func_objetivo(pop[individuo]):
            proxPop[individuo] = novoIndividuo
        else:
            proxPop[individuo] = pop[individuo]
    pop = proxPop

print(pop)
for i in pop:
    print(func_objetivo(i))