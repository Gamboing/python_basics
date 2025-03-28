arr3=[]
arr3=input("Ingresa el modelo de tu ecuacion: ")
arr4=[40,40,97,43,98,41,42,121,41]
x=arr4
z=arr3.split()
n=len(z)
Asc=0
for i in range (n):
    if ord(z[i]) == x[i]:
        z="".join(z)

    else:
        z.insert(i,x[i])
        z="".join(z)   
print(z)

#40
#40
#97
#43
#98
#41
#42
#121
#41