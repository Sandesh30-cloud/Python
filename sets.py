# harry = {}        #dictionary
# print(type(harry))

s1 = {1,2,3,5}      #set s1 U s2
s2 = {5,6,7,8}
print(s1.union(s2))
print(type(s1))

cities = {"tokyo","mumbai","dubai","berlin"}
cities2 = {"tokyo","nicaragoa","ertrai","dubai"}
cities3 = cities.union(cities2)
print(cities3)
cities4 = cities.intersection(cities2)
print(cities4)
cities5 = cities.difference(cities2)
print(cities5)