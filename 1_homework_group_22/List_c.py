list = [x**2 for x in range(101)]
print("List: {}\n".format(list))
new_list = [x for x in list if x%2 == 0]
print("New list: {}".format(new_list))