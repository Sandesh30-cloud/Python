countries = ("spain","Italy","Russia","England")
temp = list(countries)
temp.append("Germany")    #add item
temp.pop(3)               #remove item
temp[2] = "Finland"       #change item
countries = tuple(temp)
print(countries)