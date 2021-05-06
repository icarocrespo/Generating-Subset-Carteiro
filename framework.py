#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Rodrigo Mansilha
# Universidade Federal do Pampa (Unipampa)
# Programa de Pós-Graduação em Eng. de Software (PPGES)
# Bacharelado em: Ciência da Camputação, Eng. de Software, Eng. de Telecomunicações

# Algoritmos
# Laboratório 1: apresentação de resultados experimentais e analíticos

try:
    import sys
    import os
    import argparse
    import logging
    import subprocess

    from scipy.special import factorial
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.optimize as opt
    import matplotlib.colors as colors
    import matplotlib.cm as cmx


except ImportError as error:
    print(error)
    print()
    print("1. (optional) Setup a virtual environment: ")
    print("  python3 -m venv ~/Python3env/algoritmos ")
    print("  source ~/Python3env/algoritmos/bin/activate ")
    print()
    print("2. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

# Lista completa de mapas de cores
# https://matplotlib.org/examples/color/colormaps_reference.html
mapa_cor = plt.get_cmap('tab20')  # carrega tabela de cores conforme dicionário
mapeamento_normalizado = colors.Normalize(vmin=0, vmax=19)  # mapeamento em 20 cores
mapa_escalar = cmx.ScalarMappable(norm=mapeamento_normalizado, cmap=mapa_cor)  # lista de cores final

formatos = ['.', 'v', 'o', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h']


# https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.plot.html
# '.'	point marker
# ','	pixel marker
# 'o'	circle marker
# 'v'	triangle_down marker
# '^'	triangle_up marker
# '<'	triangle_left marker
# '>'	triangle_right marker
# '1'	tri_down marker
# '2'	tri_up marker
# '3'	tri_left marker
# '4'	tri_right marker
# 's'	square marker
# 'p'	pentagon marker
# '*'	star marker
# 'h'	hexagon1 marker
# 'H'	hexagon2 marker
# '+'	plus marker
# 'x'	x marker
# 'D'	diamond marker
# 'd'	thin_diamond marker
# '|'	vline marker
# '_'	hline marker


def carrega_arquivo(nome_arquivo):
    '''
	Carrega dados de um arquivo para memória
	:param nome_arquivo: a ser carregado
	:return: n, medias, desvios
	'''
    f = open(nome_arquivo, "r")
    n = []
    medias = []
    desvios = []
    for l in f:
        print("linha: {}".format(l))
        if l[0] != "#":
            n.append(int(l.split(" ")[0]))
            medias.append(float(l.split(" ")[1]))
            desvios.append(float(l.split(" ")[2]))
    f.close()
    return n, medias, desvios


def funcao_linear(n, cpu):
    '''
	Aproximação fatorial
	:param n: tamanho da instância
	:param cpu: fator de conversão para tempo de CPU
	:return: aproximação
	'''
    return (n * cpu)

def funcao_constante(n, cpu):
    return 2 * cpu


def funcao_quadratica(n, cpu):
    '''
	Aproximação quadrática
	:param n: tamanho da instância
	:param cpu: fator de conversão para tempo de CPU
	:return: aproximação
	'''
    return (n * n * cpu)


def funcao_cubica(n, cpu):
    '''
	Aproximação quadrática
	:param n: tamanho da instância
	:param cpu: fator de conversão para tempo de CPU
	:return: aproximação
	'''
    return (n * n * n * cpu)

def funcao_exponencial(n, cpu):
    '''
	Aproximação quadrática
	:param n: tamanho da instância
	:param cpu: fator de conversão para tempo de CPU
	:return: aproximação
	'''
    return (np.exp(n) * cpu)


def funcao_fatorial(n, cpu):
    '''
	Aproximação linear
	:param n: tamanho da instância
	:param cpu: fator de conversão para tempo de CPU
	:return: aproximação
	'''
    return (factorial(n) * cpu)

def main():
    '''
	Programa principal
	:return:
	'''
    # carrega dados do arquivo
    tamanhos, medias, desvios = carrega_arquivo("out_numero_bairros.txt")

    # realiza aproximação
    parametros, pcov = opt.curve_fit(funcao_constante, xdata=tamanhos, ydata=medias)
    aproximados = [funcao_constante(x, *parametros) for x in tamanhos]
    print("aproximados:           {}".format(aproximados))
    print("parametros_otimizados: {}".format(parametros))
    print("pcov:                  {}".format(pcov))

    # mostra dados
    print("Tamanho\tMedia\t\tDesvio\t\tAproximado")
    for i in range(len(tamanhos)):
        print("%03d\t%02f\t%02f\t%02f" % (tamanhos[i], medias[i], desvios[i], aproximados[i]))
    print("")

    # plota aproximação
    curva = "Carteiro aproximado"
    cor_rgb = mapa_escalar.to_rgba(1)
    plt.plot(tamanhos, aproximados, label=curva, color=cor_rgb)

    # plota medições
    curva = "Carteiro Medido"
    cor_rgb = mapa_escalar.to_rgba(0)
    plt.errorbar(x=tamanhos, y=medias, yerr=desvios, fmt=formatos[1], label=curva, linewidth=2, color=cor_rgb)

    # configurações gerais
    plt.legend()
    plt.xticks(tamanhos)
    plt.title("Tempo de execução de annealing p/ Carteiro (3 tentativas)")
    plt.xlabel("Quantidade da Bairros")
    plt.ylabel("Tempo de execução (s)")

    # mostra
    # plt.show()

    # salva em png
    plt.savefig('out.png', dpi=300)


if __name__ == '__main__':
    sys.exit(main())
