from time import *
import random as rd

def mistakes(para, typed):
    count = 0
    for i in range(len(para)):
        try:
            if para[i] != typed[i]:
                count += 1
        except:
            count += 1
    return count


para = ["Hello my dear friends this is the worst place to get education in the world",
        "Welcome to the world of Science and Technology",
        "Welcome to the world of Space, galaxies, stars, planets and moons "]
test1 = rd.choice(para)

print("Here is the paragraph you have to type: ")
print(test1)
print(end="\n")

print("Start typing : ")
start = time()
typed = input()
end = time()
total_time = int(end - start)

print(f"Time taken to type the paragraph: {total_time} w/sec")
speed = len(typed)/total_time
print(f"Your typing speed is: {speed} words")
print(f"Number of mistakes you made: {mistakes(test1, typed)}")
