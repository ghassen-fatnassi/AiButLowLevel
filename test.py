n=5
anss=0

def boocle(n):
    global anss
    if n==0:
        anss+=1
    for i in range(n):
        boocle(n)
boocle(n)
print(anss)