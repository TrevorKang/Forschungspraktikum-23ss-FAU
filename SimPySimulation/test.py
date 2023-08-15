import random

numbers = [1, 2, 3, 4, 5]
probabilities = [0.1, 0.3, 0.4, 0.1, 0.1]

# Generate a random number based on the probabilities
# for i in range(100):
#     selected_number = random.choices(numbers, probabilities)[0]
#
#     print("Generated number:", selected_number)


waiting_time_au = {"1": [], "2": [], "3": [], "4": [], "5": []}

p1 = 25
waiting_time_au.get(str(1)).append(p1)
print(waiting_time_au)