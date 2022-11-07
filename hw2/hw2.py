import sys
import math


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    # Implementing vectors e,s as lists (arrays) of length 26
    # with p[0] being the probability of 'A' and so on
    e = [0] * 26
    s = [0] * 26

    with open('e.txt', encoding='utf-8') as f:
        for line in f:
            # strip: removes the newline character
            # split: split the string on space character
            char, prob = line.strip().split(" ")
            # ord('E') gives the ASCII (integer) value of character 'E'
            # we then subtract it from 'A' to give array index
            # This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char) - ord('A')] = float(prob)
    f.close()

    with open('s.txt', encoding='utf-8') as f:
        for line in f:
            char, prob = line.strip().split(" ")
            s[ord(char) - ord('A')] = float(prob)
    f.close()

    return (e, s)


def shred(filename):
    # Using a dictionary here. You may change this to any data structure of
    # your choice such as lists (X=[]) etc. for the assignment
    X = dict()
    for i in range(65, 91):
        X[chr(i)] = 0

    with open(filename, encoding='utf-8') as f:
        while True:
            c = f.read(1)
            if not c:
                break

            c = c.upper()
            if len(c) == 0 or ord(c) < 65 or ord(c) > 90:
                continue

            X[c] = X[c] + 1

    return X


e, s = get_parameter_vectors()
x = shred("samples/letter0.txt")
print("Q1")
# print(x)
for letter in x:
    print(letter, x[letter])
print("Q2")
print("%.4f" % (x['A'] * math.log(e[0])))
print("%.4f" % (x['A'] * math.log(s[0])))
print("Q3")
sum_e = 0
index = 0
for letter in x:
    sum_e += x[letter] * math.log(e[index])
    index += 1
print("%.4f" % (math.log(0.6) + sum_e))
sum_s = 0
index = 0
for letter in x:
    sum_s += x[letter] * math.log(s[index])
    index += 1
print("%.4f" % (math.log(0.4) + sum_s))
print("Q4")
if (math.log(0.4) + sum_s) - (math.log(0.6) + sum_e) >= 100:  # spanish - english >= 100
    print(1.0000)
elif (math.log(0.4) + sum_s) - (math.log(0.6) + sum_e) <= -100:  # spanish - english <= -100
    print(0.0000)
else:
    denom = 1 + math.e ** ((math.log(0.4) + sum_s) - (math.log(0.6) + sum_e))  # 1 + e^(F(spanish)-F(english))
    print("%.4f" % (1 / denom))
