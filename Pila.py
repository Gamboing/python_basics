arr2=["(", "(", "a", "+", "b", ")", "*", "y", ")"]
arr3=[]
arr3=input("Ingresa el modelo de tu ecuacion: ")
z=arr3.split()
n=len(z)
m=len(arr2)


if n<m:
    z.append("")
elif n>m:
    z.pop()

for i in range(n):
    if arr2[i] == z[i]:
        d="".join(z)
    else :
        z.insert(i,arr2[i])
        d="".join(z) 
print(d)
#print (f"La ecuación es correcta{d}")