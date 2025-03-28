arr=[]
arr = input("Ingresa el modelo de tu ecuacion: ")
x = arr.split()
n=len(x)-1

if x[0] == "(" and x[n] == ")":
    x= "".join(x)
    print(f"La ecuacion es correcta{x}")
elif x[0] == "(":
    x.append(")")
    z="".join(x)
    print(f"La ecuación a la que te referías es así= {z}")

#################################################################


