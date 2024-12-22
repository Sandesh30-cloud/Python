n = int(input("Enter number : "))
def square(n):
    '''takes in a number n,returns
    the square of n '''
    print(n**2)
    square(5)
    print(square.__doc__)