import random
maximo=0
# Inicializamos el capital
C = [20]  # Lista que contiene el capital inicial

# Establecemos la condición para que el capital nunca se vuelva negativo
aux = C[0]

while aux > 0:  # Mientras no me haya arruinado
    aux = aux + 2 * (random.random() < 0.5) - 1  # Cambiamos el capital
    C.append(aux)  # Añadimos el nuevo valor de capital
    
print(C)
maximo=max(C)
print(maximo)

#Ahora de todo ese arreglo necesito hacer que me marque el momento en el que mas dinero tuvo 
#Para eso necesito encontrar el valor maximo de la lista y su posicion
