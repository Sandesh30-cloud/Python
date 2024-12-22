questions = [
  [
    "Which language was used to create fb?", "Python", "French", "JavaScript",
    "Php", "None", 4
  ],
  [
    "Which language is most used in gaming?", "Python", "Hindi",
    "C++","JavaScript", "None", 3
  ],
  [
    "Who is the founder of Microsoft?", "Bill Gates", "Larry Page",
    "Amit Shah","Khandu Don", "None", 1
  ],
   [
    "Which region is called the roof of the world ?", "Arctic Region", "Mount Everest",
    "Siberia","Tibet", "None", 4
  ],
    [
    "Which is called the Emerald Island ?", "Sicily", "Gibralta",
    "Britain"," Ireland", "None", 4
  ],
     [
    "Which country has a highly developed dairy industry ?", "Netherlands", "Denmark",
    "France","Germany", "None", 2
  ],
     [
    "Which of the following used to be considered a buffer state ?", "Switzerland", "France", "West Germany",
    "Belgium", "None", 1
  ],
     [
    "Which country is called the Great Britain of the East ?", "Japan", "India", "Korea",
    "Russia", "None", 1
  ]
]

levels = [1000, 2000, 3000, 5000, 10000, 20000, 40000, 80000, 160000, 320000, 640000, 1280000, 2500000]
money = 0
for i in range(0, len(questions)):
  
  question = questions[i]
  print(f"\n\nQuestion for Rs. {levels[i]}")
  print(f"a. {question[1]}          b. {question[2]} ")
  print(f"c. {question[3]}          d. {question[4]} ")
  reply = int(input("Enter your answer (1-4) or  0 to quit:\n" ))
  if (reply == 0):
    money = levels[i-1]
    break
  if(reply == question[-1]):
    print(f"Correct answer, you have won Rs. {levels[i]}")
    if(i == 4):
      money = 10000
    elif(i == 9):
      money = 320000
    elif(i == 14):
      money = 10000000
  else:
    print("Wrong answer!")
    break 

print(f"You won Rs.{money}")
