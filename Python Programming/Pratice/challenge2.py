# Allow the user to enter an integer number. Determine if the number is prime
# or not. A number is prime if it is only divisible exactly by 1 and itself,
# e.g. 3, 7, 11 and 17. Note, 2 is the only even prime number. 1 is not
# generally regarded as prime.
# Display the answer “prime” or not prime as appropriate.

  # boolean data type - note the capital 'T' in 'True
def prime_number(x):
    det = 0
    game = 0
    while game == 0 :
        for i in (range(1, x+1)):
            if x % i == 0:
                det += 1
            game = 1
    if det == 2 :
        print(True)
    elif x < 2 :
        print(False)
    else:
        print(False)


x = int(input('Enter the number to test: '))
prime_number(x)
