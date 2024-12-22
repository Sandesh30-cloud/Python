def factorial(n):
    if(n==0 or n==1):
        return 1
    else:
        return n*factorial(n-1)
    
print("_________Fibonnachi Number_________")
n=input("Enter the number : ")
print("The factorial of",n,"is",factorial(int(n)))