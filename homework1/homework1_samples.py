"""Sample solutions from various students"""

## Exercise 1

# Martin's solution:
def vowelcount(s):
    """Count vowels in s"""
    s = s.lower()
    nv = 0
    for v in 'aeiou':
        nv += s.count(v)
    return nv


# Student solution:
def vowelcount(thing):
    """Returns number of vowels in thing"""
    count = 0
    vowels = "AaEeIiOoUu"
    for vowel in vowels:
        count += thing.count(vowel)
    return count


# Student solution:
def vowelcount(s):
    """
    counts the nummber of vowels in a string s
    """
    count = 0
    v = ['a','e','i','o','u']
    s = s.lower()
    for i in range(len(v)):
        count += s.count(v[i])
    return count


# Student solution:
def vowelcount(string):
    vowels = 0
    for char in string:
        if char in "aeiouAEIOU":
            vowels = vowels + 1
    return vowels


# Student solution:
def vowelcount(x):
    """Return number of vowels for a given string (regardless of case)"""
    a = 'aeiou'
    count = 0
    for i in range(len(x)):
        for j in range(len(a)):
            if x[i].lower() == a[j]:
                count = count + 1
    return count


# Student solution:
def vowelcount(s):
    """Return the number of vowels in a string"""
    v = ['a', 'e', 'i', 'o', 'u']
    count = 0
    for l in s:
        if l in v:
            count += 1
    return count


# Student solution:
VOWELS = 'aeiou'
def vowelcount(word):
    ''' Takes a str as an argument and returns the number of vowels '''
    word = word.lower()  # to count upper case vowels too
    #using list comp for brevity
    return sum(word.count(vowel) for vowel in VOWELS)


# Student solution:
def vowelcount(my_string):
    """counts the numbers of vowels (a,e,i,o,u) in a given string"""
    my_string = my_string.lower()
    count = 0
    for ii in my_string:
        if ii == 'a' or ii == 'e' or ii == 'i' or ii == 'o' or ii == 'u':
            count += 1
    return count


# Student solution:
def vowelcount(string):
    """returns number of vowels in a string"""
    l = string.lower()
    a = l.count('a')
    e = l.count('e')
    i = l.count('i')
    o = l.count('o')
    u = l.count('u')
    return a, e, i, o, u


# Student solution:
def vowelcount(string):
    """Take a string, returns number of vowels"""
    s2 = string.lower()
    new = s2.replace('a','x').replace('e','x').replace('i','x').replace('o','x').replace('u','x')
    result = new.count('x')
    return result


## Exercise 2

# Martin's solution:
def metric(x, y):
    """Calculate difference over sum"""
    d = x - y
    s = x + y
    print('difference is %g, sum is %g' % (d, s))
    # optional: handle divide by zero:
    if s == 0:
        return None
    return d / s


# Student solution:
def metric(x, y):
    s = x + y
    d = x - y
    print('difference is %d, sum is %d' % (x - y, x + y))
    if x + y == 0:
        print('division by zero not possible')
        return
    return (x - y)/(x + y)


# Student solution:
def metric(x, y):
    """this function prints the difference and sum of x and y and returns diff/sum"""
    sum = x + y
    diff = x - y
    if type(sum) == float and type(diff) == float:
        print('sum is %f and difference is %f' %(sum, diff))
    else:
        print('sum is %i and difference is %i' %(sum, diff))
    if sum == 0:
        print('output error: sum is zero')
        return None
    else:
        return diff / sum


# Student solution:
def metric(x,y):
    """calculates difference, sum and difference/sum"""
    difference = x - y
    sum = x + y
    print("difference is %f" % difference)
    print("sum is %f" % sum)
    return 0 if sum == 0 else difference / sum


# Student solution:
def metric(x, y):
    """Prints difference, sum  auf x and y and returns difference/sum"""
    sum1 = x + y
    dif = x - y
    if sum1 != 0:
        divide = dif // sum1
    else:
        divide = 'N/A, sum is zero'
    return 'difference is ', dif, 'sum is', sum1, '; difference/sum=', divide


# Student solution:
def metric(x, y):
    """Returns difference and sum x and y and the difference devided by the sum"""
    d = x - y
    s = x + y
    if s != 0:
        a = str(d / s)
    elif s == 0:
        a = 'invalid'
    return str('Difference is %.3f' % d), str('Sum is %.3f' % s), str('Difference devided by sum is %s' % a)

## Exercise 3

# Martin's solution:
def multtable(n):
    """Print multiplication table from 1 to n"""
    for i in range(1, n+1): # rows
        for j in range(1, n+1): # columns
            print(i * j, end=' ')
        print() # move to new line for next row


# Student solution:
def multtable(n):
    """takes a number n and prints out the multiplication table for integers 1 through n"""
    for ii in range(1, n+1):
        print('\n')
        for jj in range(1, n+1):
            print(ii*jj, end = '\t')


# Student solution:
def multtable(n):
    """ This function takes a number n as an argument and prints out the multiplication table
    for integers 1 through n. """
    for i in range(n):
        row = []
        for j in range(n+1):
            num = (i + 1) * (j + 1)
            row.append(num)
        print(row[:n], sep = ',', end = '\n')
