# Allow the user to enter a number. Use an if statement to display
# whether the number is negative, equal to zero or positive.
number = int(input('Enter your number: '))
if number > 0:
    print('the number is positive value')
elif number == 0 :
    print('the number is zero')
else: print('the number is negative')



# The range() function can be used to create a sequence of numbers
# (see the code for this). Allow the user to enter a number and
# then iterate over the range to determine whether the number
# the user entered is present in the sequence.
# Note: the returned range does not include the end value.
# See the documentation for more details: 
# https://docs.python.org/3/library/functions.html#func-range

x1 = range(0, 20)
x2 = range(0, 30, 2)
x3 = range (20, 1, -3)

num = int(input('Please enter a number: '))
for i in x1:
    if i == num:
        print('Number in generated range sequence')
        break

for i in x2:
    if i == num:
        print('Number in generated range sequence')
        break

for i in x3:
    if i == num:
        print('Number in generated range sequence')
        break

# This can be done more simply and pythonicly using the "in" operator e.g.:
if num in x1:
    print('Number in generated range sequence')


# An arithmetic series of numbers is defined by an initial value
# and a common difference. For example, if the starting value is 2 and
# the common difference is 5, the arithmetic sequence would begin:
# 2, 5, 8, 11, 14, ...
# Allow the user to enter values for the initial value and common difference.
# Write code to determine the sum of the first 10 elements in the
# arithmetic sequence.

initial_value = int(input('Enter initial value: '))
common_diff = int(input('Enter common difference: '))
gap = abs(initial_value - common_diff)
# Obvious approach - use a loop
x = initial_value
list1 = []
list1.append(initial_value)
list1.append(x)
for i in range(0,10):
    if len(list1) != 10:
        x += gap
        list1.append(x)
    elif len(list1) == 10:
        sum_value = sum(list1)
        print(list1)
        print(f'Your sum of 10 elements is {sum_value}')
        break
    else: break

# Alternative approach - use a range...
x = range(initial_value, initial_value + (10 * common_diff), common_diff)
print(f'Sum is {sum(x)}')


# Write code to simulate the simple children’s game “Fuzz Buzz”.
# The rules are simple; start counting upwards from 1 in single increments.
# If the number is divisible by 5, display “Fuzz”. if the number is
# divisible by 6, display “Buzz". If the number is divisible by both,
# display “Fuzz Buzz”. Otherwise just display the number.
# Play the game from 1 through to 50 inclusive.

v1 = 1   # current number value
count_1 = 0
while v1 <= 49:
    if v1 % 5 == 0 and v1 % 6 != 0:
        v1 += 1
        print('Fizz')
    elif v1 % 5 != 0 and v1 % 6 == 0:
        v1 += 1
        print('Buzz')
    elif v1 % 5 == 0 and v1 % 6 == 0:
        v1 += 1
        print('FizzBuzz')
    else:
        v1 += 1
        print(v1)
