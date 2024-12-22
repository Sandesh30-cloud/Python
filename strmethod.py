# strings are immutable
a="sandesh"
print(a.capitalize()) # 1st letter capital
print(a.isupper())
print(a.upper())
b="computer !!!"
print(b.strip("!"))
print(b.endswith("!"))
print(b.replace("computer","Engineering"))

str1="Hello I am trader in forex"
print(str1.find("trader")) #returns index no. if wor is there
print(str1.find("traderss"))# returns -1
print(str1.isalnum())# A-Z ,a-z, 0-9
str2="Hello201"
print(str2.isalnum())