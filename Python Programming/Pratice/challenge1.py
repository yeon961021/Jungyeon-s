# Generate a sequence of integer numbers in the pattern 0, 2, 4, ..., up to 20
# inclusive and display the numbers on the screen.
x = 0
limit = 22
# Your solution here...
while x < limit :
    print(x)
    x += 2


# Allow the user to enter a string, and then print out each letter
# in the string one at a time on a separate line.
# Your solution here...
string = input('Enter some text: ')
number = len(string)
x = 0
while x < number:
    print(string[x])
    x += 1

# As above, but print out the letters in the string in reverse order.
# Your solution here...
string = input('Enter some text: ')
number = -(len(string)) -1
x = -1
while x != number:
    print(string[x])
    x += -1

# Starting with the initial value 0 and 1, generate a Fibonacci sequence.
# Each element in a Fibonacci sequence is the sum of the two previous
# elements e.g. 0, 1, 1, 2, 3, 5, 8, ...
# Allow the user to specify how many elements should be generated.

x = int(input('How many elements? '))
y = [0,1]
i = 0
while len(y) < x :
    t = y[i] + y[i+1]
    i += 1
    y.append(t)
print(y)

# Your solution here...
