arr = [[11, 2, 4], [4, 5, 6], [10, 8, -12]]
total=0
n = len(arr)  
suma1 = 0
suma2 = 0

for i in range(n):
    suma1 += arr[i][i]         
    suma2 += arr[i][n-1-i]     
        
    
total=suma1-suma2
print(abs(total))

