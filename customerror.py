a = int(input("Enter the no. :"))

if(a<5 or a>9):
    raise ValueError("Provided value is not in range")
else:
    print(a)
    