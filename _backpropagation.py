import numpy as np

def funcionAproximar(p):
	return (1 + np.sin((np.pi/4)*p))

p = np.array(([1]), dtype=float)
g = funcionAproximar(1)
t = np.array(([g]), dtype=float)

numEntradas = 1
numSalidas = 1
numOcultas = 2

#Pesos de entrada a la capa oculta
#W1 = np.random.randn(numEntradas, numOcultas)
#Pesos de capa oculta a salida
#W2 = np.random.randn(numOcultas, numSalidas)

#Bias de entrada a la capa oculta
#bias1 = np.random.randn(numEntradas, numOcultas)
#Bias de capa oculta a salida
#bias2 = np.random.randn(numSalidas)

#Pesos de entrada a la capa oculta
W1 = np.array([[-0.27,-0.41]])
#Pesos de capa oculta a salida
W2 = np.array([[0.09],[-0.17]])

#Bias de entrada a la capa oculta
bias1 = np.array([[-0.48,-0.13]])
#Bias de capa oculta a salida
bias2 = np.array([[0.48]])

z2 = 0.0

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

	tasa_aprendizaje = 0.1

	a_error = t - a
	sensibilidad_2 = -2 * derivadaPureline(a) * a_error
	sensibilidad_1 = derivadaLogSigmoide(a1) * sensibilidad_2.dot(W2.T)	

	W1 -= tasa_aprendizaje * p.T.dot(sensibilidad_1)
	W2 -= tasa_aprendizaje * a1.T.dot(sensibilidad_2)

	bias1 -= tasa_aprendizaje * sensibilidad_1
	bias2 -= tasa_aprendizaje * sensibilidad_2

def iniciarAlgoritmo(p,t):
    a = forward(p)
    backward(p,t,a)

iteraciones_max = 1000

print("Entrada: \n" + str(p))
print("Salida esperada: \n" + str(t))
print("\n")

for i in range(iteraciones_max):	
	iniciarAlgoritmo(p,t)

print("W1: \n" + str(W1))
print("bias1: \n" + str(bias1))
print("\n")
print("W2: \n" + str(W2))
print("bias2: \n" + str(bias2))