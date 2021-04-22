from math import sqrt
import random, decimal

print("Problema do Carteiro")

def entrada_tempo():
    tempo = input("Valor do Tempo:")
    return float(tempo)

def entrada_alfa():
    print("OBS: O Alfa deve ser menor que 1.")
    alfa = input("Valor do Alfa:")
    if(float(alfa) >= 1.0):
        print("Alfa maior que um. Por favor insira um valor menor.")
        entrada_alfa()

    return float(alfa)


f = open("bairros 16","r")
l_bairros = f.read().split()
f.close()
bairros = {} #dicionario com a posicao de todos os bairros {'bairro_numero':[posX, posY]}

for x in range(0,len(l_bairros),3): #preenche o dicionario com os bairros e posicoes
    bairros[l_bairros[x]] = [float(l_bairros[x+1]),float(l_bairros[x+2])]


def distancia(xyA,xyB): #calcula a distancia reta entre dois pontos
    xA, xB, yA, yB = (xyA[0]), (xyB[0]), (xyA[1]), (xyB[1])
    d = sqrt((xB-xA)**2 + (yB-yA)**2)
    return round(d,12)

bairros_custo = {} #dicionario com o custo de cada travessia {('bairroA_numero','bairroB_numero'): distancia}
for k in range(1,17):
    for c in range(1,17):
        bairros_custo[(str(k),str(c))] = distancia(bairros[str(k)],bairros[str(c)])

def custo_total(lista_bairros): #retorna o custo total de uma solucao
    custo = 0
    for bairro in range(len(lista_bairros)):
        if bairro == len(lista_bairros)-1: #se chegou no ultimo bairro soma o custo com a origem
            custo += bairros_custo[(str(lista_bairros[bairro]), str(lista_bairros[0]))]
        else:
            custo+=bairros_custo[(str(lista_bairros[bairro]),str(lista_bairros[bairro+1]))]
    return custo

def vizinho(solucao):
    solucao_anterior = solucao.copy()
    while True:
        posA = random.randint(0,15)
        posB = random.randint(0,15)
        a = solucao[posA]
        b = solucao[posB]
        solucao[posA] = b
        solucao[posB] = a

        posC = random.randint(0, 15)
        posD = random.randint(0, 15)
        c = solucao[posC]
        d = solucao[posD]
        solucao[posC] = d
        solucao[posD] = c
        if solucao != solucao_anterior:
            break
    return solucao

def probabilidade(custo_antigo,custo_novo,temperatura): #calcula a probabilidade de aceitacao da nova solucao
    decimal.getcontext().prec = 100
    diferenca_custo = custo_antigo - custo_novo
    custo_temp = diferenca_custo/temperatura
    p = decimal.Decimal(0)
    e = decimal.Decimal(2.71828)
    n_custo_temp = decimal.Decimal(-custo_temp)
    try:
        p = e**n_custo_temp
        resultado = repr(p)
    except decimal.Overflow:
        #print("Error decimal Overflow")
        return 0.0

    try: #caso o numero tenha casas decimais
        fim = resultado.find("')")
        resultado = round(float(resultado[9:fim-1]), 3)

    except: #numero n tem casas decimais
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

        tempo = tempo*alfa
    return melhor_solucao, melhor_custo

def gerar_solucao(): #gera uma solucao aleatoria
    solucao_aleatoria = [x for x in range(1,17)]
    random.shuffle(solucao_aleatoria)
    return solucao_aleatoria

solucao_inicial = gerar_solucao()

solucao_final, custo = annealing(solucao_inicial, entrada_tempo(), entrada_alfa())
print(solucao_final, "Solução Final \n", custo,"Custo Final")
