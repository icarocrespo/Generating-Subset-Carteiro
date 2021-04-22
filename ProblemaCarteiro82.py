DEFAULT_OUT = "problema_carteiro_82.txt"
DEFAULT_SEED = None

DEFAULT_N_START = 1
DEFAULT_N_STOP = 10
DEFAULT_N_STEP = 1
DEFAULT_TRIALS = 3

from subprocess import Popen, PIPE
from time import sleep, time
from multiprocessing import Process
import shlex
import json


import sys
import os
import argparse
import logging
import subprocess

from math import sqrt
import random, decimal
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import matplotlib.colors as colors
import matplotlib.cm as cmx

import timeit

print("Problema do Carteiro")


def entrada_tempo():
    tempo = input("Valor do Tempo:")
    return float(tempo)


def entrada_alfa():
    print("OBS: O Alfa deve ser menor que 1.")
    alfa = input("Valor do Alfa:")
    if (float(alfa) >= 1.0):
        print("Alfa maior que um. Por favor insira um valor menor.")
        entrada_alfa()

    return float(alfa)


f = open("bairros 82", "r")
l_bairros = f.read().split()
f.close()
bairros = {}  # dicionario com a posicao de todos os bairros {'bairro_numero':[posX, posY]}

for x in range(0, len(l_bairros), 3):  # preenche o dicionario com os bairros e posicoes
    bairros[l_bairros[x]] = [float(l_bairros[x + 1]), float(l_bairros[x + 2])]


def distancia(xyA, xyB):  # calcula a distancia reta entre dois pontos
    xA, xB, yA, yB = (xyA[0]), (xyB[0]), (xyA[1]), (xyB[1])
    d = sqrt((xB - xA) ** 2 + (yB - yA) ** 2)
    return round(d, 12)


bairros_custo = {}  # dicionario com o custo de cada travessia {('bairroA_numero','bairroB_numero'): distancia}
for k in range(1, 83):
    for c in range(1, 83):
        bairros_custo[(str(k), str(c))] = distancia(bairros[str(k)], bairros[str(c)])


def custo_total(lista_bairros):  # retorna o custo total de uma solucao
    custo = 0
    for bairro in range(len(lista_bairros)):
        if bairro == len(lista_bairros) - 1:  # se chegou no ultimo bairro soma o custo com a origem
            custo += bairros_custo[(str(lista_bairros[bairro]), str(lista_bairros[0]))]
        else:
            custo += bairros_custo[(str(lista_bairros[bairro]), str(lista_bairros[bairro + 1]))]
    return custo


def vizinho(solucao):
    solucao_anterior = solucao.copy()
    while True:
        posA = random.randint(0, 81)
        posB = random.randint(0, 81)
        a = solucao[posA]
        b = solucao[posB]
        solucao[posA] = b
        solucao[posB] = a

        posC = random.randint(0, 81)
        posD = random.randint(0, 81)
        c = solucao[posC]
        d = solucao[posD]
        solucao[posC] = d
        solucao[posD] = c
        if solucao != solucao_anterior:
            break
    return solucao


def probabilidade(custo_antigo, custo_novo, temperatura):  # calcula a probabilidade de aceitacao da nova solucao
    decimal.getcontext().prec = 100
    diferenca_custo = custo_antigo - custo_novo
    custo_temp = diferenca_custo / temperatura
    p = decimal.Decimal(0)
    e = decimal.Decimal(2.71828)
    n_custo_temp = decimal.Decimal(-custo_temp)
    try:
        p = e ** n_custo_temp
        resultado = repr(p)
    except decimal.Overflow:
        # print("Error decimal Overflow")
        return 0.0

    try:  # caso o numero tenha casas decimais
        fim = resultado.find("')")
        resultado = round(float(resultado[9:fim - 1]), 3)

    except:  # numero n tem casas decimais
        resultado = round(float(resultado[9:-2]))
    return resultado


def annealing(solucao, tempo, alfa):
    print("Calculando rotas.....\n")
    custo_antigo = custo_total(solucao)
    # tempo = entrada_tempo()
    tempo_minimo = 0.0001
    #  alfa = entrada_alfa()
    melhor_solucao, melhor_custo = solucao[::], custo_antigo
    while tempo > tempo_minimo:
        i = 1
        while i <= 500:
            nova_solucao = vizinho(solucao)
            novo_custo = custo_total(nova_solucao)
            p = probabilidade(custo_antigo, novo_custo, tempo)
            if novo_custo < melhor_custo:
                melhor_solucao = nova_solucao[::]
                melhor_custo = novo_custo
            if p > round(random.random(), 3):
                solucao = nova_solucao[::]
                custo_antigo = novo_custo
            i += 1

        tempo = tempo * alfa
    return melhor_solucao, melhor_custo


def gerar_solucao():  # gera uma solucao aleatoria
    solucao_aleatoria = [x for x in range(1, 83)]
    random.shuffle(solucao_aleatoria)
    return solucao_aleatoria


def problema_carteiro():
    solucao_inicial = gerar_solucao()

    solucao_final, custo = annealing(solucao_inicial, 1.0, 0.9)
    print(solucao_final, "Solução Final \n", custo, "Custo Final")


def main():
    # Definição de argumentos
    parser = argparse.ArgumentParser(description='Problema Carteiro N=82')
    help_msg = "arquivo de saída.  Padrão:{}".format(DEFAULT_OUT)
    parser.add_argument("--out", "-o", help=help_msg, default=DEFAULT_OUT, type=str)

    help_msg = "semente aleatória. Padrão:{}".format(DEFAULT_SEED)
    parser.add_argument("--seed", "-s", help=help_msg, default=DEFAULT_SEED, type=int)

    help_msg = "n máximo.          Padrão:{}".format(DEFAULT_N_STOP)
    parser.add_argument("--nstop", "-n", help=help_msg, default=DEFAULT_N_STOP, type=int)

    help_msg = "n mínimo.          Padrão:{}".format(DEFAULT_N_START)
    parser.add_argument("--nstart", "-a", help=help_msg, default=DEFAULT_N_START, type=int)

    help_msg = "n passo.           Padrão:{}".format(DEFAULT_N_STEP)
    parser.add_argument("--nstep", "-e", help=help_msg, default=DEFAULT_N_STEP, type=int)

    help_msg = "tentativas.        Padrão:{}".format(DEFAULT_N_STEP)
    parser.add_argument("--trials", "-t", help=help_msg, default=DEFAULT_TRIALS, type=int)

    # Lê argumentos from da linha de comando
    args = parser.parse_args()

    trials = args.trials
    f = open(args.out, "w")
    f.write("#Problema Carteiro 82\n")
    f.write("#n time_s_avg time_s_std (for {} trials)\n".format(trials))
    m = 100
    np.random.seed(args.seed)
    for n in range(args.nstart, args.nstop + 1, args.nstep):  # range(1, 100):
        resultados = [0 for i in range(trials)]
        tempos = [0 for i in range(trials)]
        for trial in range(trials):
            print("\n-------")
            print("n: {} trial: {}".format(n, trial + 1))
            entrada = np.random.randint(0, n, n)
            print("Entrada: {}".format(entrada))
            tempo_inicio = timeit.default_timer()
            resultados[trial] = problema_carteiro()
            tempo_fim = timeit.default_timer()
            tempos[trial] = tempo_fim - tempo_inicio
            print("Saída: {}".format(resultados[trial]))
            print('Tempo: {} s'.format(tempos[trial]))
            print("")

        tempos_avg = np.average(tempos)  # calcula média
        tempos_std = np.std(a=tempos, ddof=False)  # ddof=calcula desvio padrao de uma amostra?

        f.write("{} {} {}\n".format(n, tempos_avg, tempos_std))
    f.close()


if __name__ == '__main__':
    sys.exit(main())
