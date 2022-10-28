def meow():
    a = 1
    while 1:
        yield "Meow "*a
        a*=2

counter = 1
for n in meow():
    print("{} {}".format(counter,n))
    counter+=1
    if counter == 5:
        break