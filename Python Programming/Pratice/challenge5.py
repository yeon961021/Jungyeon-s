# Create a list comprehension that displays only the even numbers in a given list.

list1 = [1, 6, 3, 4, 2, 5, 7, 18, 9]
new_list = []
for i in range(len(list1)):
    if list1[i] % 2 == 0:
        new_list.append(list1[i])
print(new_list)

new_list_list = [x for x in list1 if x % 2 == 0]
print(new_list_list)


# Create a list comprehension that produces a new list b from a given list of
# strings a consisting of only those elements in a that have 3 or fewer characters.

list2 = ['gnu', 'penguin', 'horse', 'fox', 'elephant', 'emu']
list_b = []

for i in range(len(list2)):
    if len(list2[i]) <= 3:
        list_b.append(list2[i])
print(list_b)

# Create a list comprehension that produces a new list b from a given list of
# strings a consisting of only those elements in a that begin with a specific letter.
list3 = ['gnu', 'penguin', 'horse', 'fox', 'elephant', 'emu', 'goat']
specific_letter = 'e'
list_c = []

for i in range(len(list3)):
    if list3[i][0] == specific_letter:
        list_c.append(list3[i])
print(list_c)

# Create a list comprehension that produces a new list b from a given list of
# strings a consisting of only those elements in a that begin with a specific letter
# and have 3 of fewer characters.
list4 = ['gnu', 'penguin', 'horse', 'fox', 'elephant', 'emu', 'goat', 'giraffe']
list_d = []
specific_letter2 = 'g'

for i in range(len(list4)):
    if list4[i][0] == specific_letter2 and len(list4[i]) <= 3:
        list_d.append(list4[i])
print(list_d)

list_e = [x for x in list4 if x[0] == specific_letter2 and len(x) <= 3]
print(list_e)