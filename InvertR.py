s="hola"
def invertir(s):
    if len (s)==0:
        return s
    return s[-1]+invertir(s[:-1]) 

print(invertir(s))




