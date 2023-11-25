import random # A library that contains random number functions and methods


def number_create(a, b):
    x = random.randint(a,b) 
    return x

def analyse_guess(guess, answer):
    if guess == answer:
        return 0
    elif guess > answer:
        return -2
    elif guess < answer:
        return -1

def check_previous(guess, list2):
    if guess in list2:
        return 0
    else:
        return 1

def attempt_check(attemps):
    if attemps == 1:
        return 0
    else:
        return 1

def main():
    answer = number_create(1, 99)
    guess_list = []
    attemps = 10
    game_mode = 1
    print("[Number Guess Game]")
    while game_mode == 1:
        game_mode = attempt_check(attemps)
        print(f'your history{guess_list}, and remaining opportunity : {attemps}')
        guess = int(input("Please enter a number (1~99, quit = -1): "))
        if guess == -1:
            game_mode = 0
            print("I'm so sad that you decided to leave this game T_T.")
            break
        dup = check_previous(guess, guess_list)
        if dup == 1:
            if guess > 99 | guess < 1:
                print('Check your number, you have to choose a number between 1 and 99')
            elif 1 <= guess and guess <= 99:
                attemps -= 1
                result = analyse_guess(guess, answer)
                if result == 0:
                    game_mode = 0
                    print('You are win!')
                    break
                elif result == -1:
                    guess_list.append(guess)
                    print('Your number is too low')
                elif result == -2:
                    guess_list.append(guess)
                    print("Your number is too high")
        elif dup == 0:
            print('Check your number, you already put this number before')
    print(f'The answer is {answer}')

main()