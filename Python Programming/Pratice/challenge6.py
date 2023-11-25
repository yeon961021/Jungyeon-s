import random as random
def gen_random(min_val, max_val):
    x = random.randrange(max_val) + min_val
    return x

gen_random(5,20)

def check_duplicate(my_list, val):
    check = 0
    while check < len(my_list):
        if my_list[check] == val:
            return True
        else:
            check += 1
    return False

def generate_ticket():
    my_list = []
    while len(my_list) < 6:
        i = gen_random(0,50)
        if check_duplicate(my_list, i) == False:
            my_list.append(i)
            my_list.sort(reverse = False)
    return my_list
"""Generate a sorted list with 6 elements in the range 1 to 50 inclusive"""

def check_ticket(my_ticket, winning_numbers):
    count = 0
    for i in range(0,6):
        if my_ticket[i] == winning_numbers[i]:
            count += 1
    if count == 2:
        print('a win of £1')
    elif count == 3:
        print('a win of £10')
    elif count == 4:
        print('a win of £50')
    elif count == 5:
        print('a win of £500')
    elif count == 6:
        print('a win of £1,000,000')
    else:
        print('Sorry, good bye!')


def main():
    my_ticket = generate_ticket()
    print(my_ticket)
    winning_numbers = generate_ticket()
    print(winning_numbers)
    check_ticket(my_ticket, winning_numbers)

main()