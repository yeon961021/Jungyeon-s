# Solution for swap sorter problem

my_list = ['gnu', 'aardvark', 'horse', 'donkey', 'eagle', 'emu']

for i2 in range(len(my_list)-1):
    for i in range(len(my_list)-1):
        if my_list[i].lower() > my_list[i+1].lower():
            swap1 = my_list[i]
            swap2 = my_list[i+1]
            my_list[i] = swap2
            my_list[i+1] = swap1
            print(f'now my_list has this shape {my_list}')
        
print(my_list)
