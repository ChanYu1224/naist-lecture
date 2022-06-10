import random
from math import gcd

# エラトステネスの篩を用いた素数判定
def is_prime(n):
    primes = []
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False

    for p in range (0, n + 1):
        if not is_prime[p]:
            continue
        primes.append(p)
        for i in range(p*p, n + 1, p):
            is_prime[i] = False

    return n in primes


# 入力
n = int(input('n : '))

# 反復回数
repetition_number = 10

# 試行
for i in range(repetition_number):
    print('---- repetition', i+1 ,'---')
    
    a = random.randint(2, n-1)
    gcd_result = gcd(a, n)
    mod_result = pow(a, n-1, n)

    print('a             :', a)
    print('gcd(a,n)      :', gcd_result)
    print('a^(n-1) mod n :', mod_result)

    print('result        : ', end='')
    if gcd_result != 1:
        print('composite number')
    elif mod_result == 1:
        print('prime number')
    else:
        print('composite number')

# 真の解
print('----------------------')
print('true answer   : ', end='')
if is_prime(n):
    print('prime number')
else:
    print('composite number')