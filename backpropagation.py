'''
	Algoritmo Backpropagation
	- Aguilar Licona Marco Antonio
	- Cortes Abraham
	FI,UNAM
'''

import numpy as np
import matplotlib.pyplot as plt

def logSigmoide(n):
	return 1/(1+np.exp(-n))

def pureline(n): 
	return n

def forward(p):
	n1 = np.dot(p,W1) + bias1
	global a1
	a1 = logSigmoide(n1)
	n2 = np.dot(a1,W2) + bias2
	a2 = pureline(n2)
	return a2

def derivadaLogSigmoide(n):
	return n * (1 - n)

def derivadaPureline(a):
	return 1

def backward(p,t,a):
	global W1
	global W2
	global bias1
	global bias2
	global aproximacion_exitosa
	global lista_errores
	global lista_salidas

	a_error = t - a

	lista_salidas.append(a[0][0])
	lista_errores.append(a_error[0][0])
	lista_targets.append(t[0])

	if a_error[0][0] == 0.0:
		aproximacion_exitosa = True

	sensibilidad_2 = -2 * derivadaPureline(a) * a_error
	sensibilidad_1 = derivadaLogSigmoide(a1) * sensibilidad_2.dot(W2.T)	

	W1 -= tasa_aprendizaje * p.T.dot(sensibilidad_1)
	W2 -= tasa_aprendizaje * a1.T.dot(sensibilidad_2)

	bias1 -= tasa_aprendizaje * sensibilidad_1
	bias2 -= tasa_aprendizaje * sensibilidad_2

def iniciarAlgoritmo(p,t):
    a = forward(p)
    backward(p,t,a)

def funcionAproximar(p,opcion):
	if opcion == 1:
		return (1 + np.sin((np.pi/4)*p))
	else:
		return (1 + np.sin((np.pi/2)*p))

num_ejercicio = 0
while (int(num_ejercicio)<1 or int(num_ejercicio)>2):
	print("\n----- Algoritmo Backpropagation -----")
	print("\nSeleccione el ejercicio para el cálculo de pesos y bias")
	print(" 1) Ejercicio 1\n 2) Ejercicio 2")
	num_ejercicio = input("\nNúmero de ejercicio: ")

	while not num_ejercicio.isdigit():
		num_ejercicio = input("\nLa entrada debe ser un número: ")

	iteraciones_max = 10000
	
	if int(num_ejercicio) == 1:
		print("\n----- Ejercicio 1 -----\n")
		p = np.array(([1]), dtype=float)
		g = funcionAproximar(1,int(num_ejercicio))
		t = np.array(([g]), dtype=float)

		aproximacion_exitosa = False
		lista_errores = []
		lista_salidas = []
		lista_targets = []
		lista_iteraciones = []

		numEntradas = 1
		numSalidas = 1
		numOcultas = 2
		tasa_aprendizaje = 0.1
		W1 = np.array([[-0.27,-0.41]])
		W2 = np.array([[0.09],[-0.17]])
		bias1 = np.array([[-0.48,-0.13]])
		bias2 = np.array([[0.48]])

		print("Entrada: \n" + str(p))
		print("Salida esperada: \n" + str(t))
		print("Tasa de aprendizaje: "+ str(tasa_aprendizaje))
		print("No. de neuronas en capa oculta: "+str(numOcultas))
		print("\n")
		iteraciones = 0
		while(aproximacion_exitosa !=True and iteraciones < iteraciones_max):	
			iniciarAlgoritmo(p,t)
			iteraciones = iteraciones + 1
			lista_iteraciones.append(iteraciones)
		print("No. de iteraciones para aproximacion exitosa: "+str(iteraciones))
		print("W1: \n" + str(W1))
		print("bias1: \n" + str(bias1))
		print("\n")
		print("W2: \n" + str(W2))
		print("bias2: \n" + str(bias2))

		plt.figure(1)
		plt.plot(lista_iteraciones,lista_salidas,'r',lista_iteraciones,lista_targets,'g')
		plt.ylabel("Salidas de la red")
		plt.xlabel("Iteraciones")
		plt.title("Comparación de función objetivo con respecto a salidas de neurona")
		
		plt.figure(2)
		plt.plot(lista_iteraciones,lista_errores)
		plt.ylabel("Error")
		plt.xlabel("Iteraciones")
		plt.title("Comportamiento de los errores en el algoritmo")

		plt.figure(3)
		plt.plot(lista_iteraciones[0:2],lista_salidas[0:2],'rs',lista_iteraciones[0:2],lista_targets[0:2],'gs')
		plt.ylabel("Salidas de la red")
		plt.xlabel("Iteraciones")
		plt.title("Comparación de función objetivo con respecto a salidas de neurona en \n2 iteraciones")
		
		plt.show()
	elif int(num_ejercicio) == 2:
		print("\n----- Ejercicio 2 -----\n")
		p = np.array(([1]), dtype=float)
		g = funcionAproximar(1,int(num_ejercicio))
		t = np.array(([g]), dtype=float)

		print("Entrada: \n" + str(p))
		print("Salida esperada: \n" + str(t))

		print("\n----- Inciso a -----")
		aproximacion_exitosa = False
		lista_errores = []
		lista_salidas = []
		lista_targets = []
		lista_iteraciones = []
		numEntradas = 1
		numSalidas = 1
		numOcultas = 2
		tasa_aprendizaje = 0.5
		W1 = np.random.uniform(-0.5,0.5,(numEntradas, numOcultas))
		W2 = np.random.uniform(-0.5,0.5,(numOcultas, numSalidas))
		bias1 = np.random.uniform(-0.5,0.5,(numEntradas, numOcultas))
		bias2 = [np.random.uniform(-0.5,0.5,numSalidas)]
		
		print("Tasa de aprendizaje: "+ str(tasa_aprendizaje))
		print("No. de neuronas en capa oculta: "+str(numOcultas))
		print("\n")
		iteraciones = 0
		while(aproximacion_exitosa !=True and iteraciones < iteraciones_max):	
			iniciarAlgoritmo(p,t)
			iteraciones = iteraciones + 1
			lista_iteraciones.append(iteraciones)
		print("No. de iteraciones para aproximacion exitosa: "+str(iteraciones))
		print("W1: \n" + str(W1))
		print("bias1: \n" + str(bias1))
		print("\n")
		print("W2: \n" + str(W2))
		print("bias2: \n" + str(bias2))

		print("\n----- Inciso b -----")
		aproximacion_exitosa = False
		lista_errores = []
		lista_salidas = []
		lista_targets = []
		lista_iteraciones = []
		numEntradas = 1
		numSalidas = 1
		numOcultas = 2
		tasa_aprendizaje = 1
		W1 = np.random.uniform(-0.5,0.5,(numEntradas, numOcultas))
		W2 = np.random.uniform(-0.5,0.5,(numOcultas, numSalidas))
		bias1 = np.random.uniform(-0.5,0.5,(numEntradas, numOcultas))
		bias2 = [np.random.uniform(-0.5,0.5,numSalidas)]
		print("Tasa de aprendizaje: "+ str(tasa_aprendizaje))
		print("No. de neuronas en capa oculta: "+str(numOcultas))
		iteraciones = 0
		while(aproximacion_exitosa !=True and iteraciones < iteraciones_max):	
			iniciarAlgoritmo(p,t)
			iteraciones = iteraciones + 1
			lista_iteraciones.append(iteraciones)
		print("No. de iteraciones para aproximacion exitosa: "+str(iteraciones))
		print("W1: \n" + str(W1))
		print("bias1: \n" + str(bias1))
		print("\n")
		print("W2: \n" + str(W2))
		print("bias2: \n" + str(bias2))

		print("\n----- Inciso c -----")
		aproximacion_exitosa = False
		lista_errores = []
		lista_salidas = []
		lista_targets = []
		lista_iteraciones = []
		numEntradas = 1
		numSalidas = 1
		numOcultas = 10
		tasa_aprendizaje = 0.5
		W1 = np.random.uniform(-0.5,0.5,(numEntradas, numOcultas))
		W2 = np.random.uniform(-0.5,0.5,(numOcultas, numSalidas))
		bias1 = np.random.uniform(-0.5,0.5,(numEntradas, numOcultas))
		bias2 = [np.random.uniform(-0.5,0.5,numSalidas)]
		print("Tasa de aprendizaje: "+ str(tasa_aprendizaje))
		print("No. de neuronas en capa oculta: "+str(numOcultas))
		print("\n")
		iteraciones = 0
		while(aproximacion_exitosa !=True and iteraciones < iteraciones_max):	
			iniciarAlgoritmo(p,t)
			iteraciones = iteraciones + 1
			lista_iteraciones.append(iteraciones)
		print("No. de iteraciones para aproximacion exitosa: "+str(iteraciones))
		print("W1: \n" + str(W1))
		print("bias1: \n" + str(bias1))
		print("\n")
		print("W2: \n" + str(W2))
		print("bias2: \n" + str(bias2))