"""
Coding exercise 2.1 from group 22

contains a constructor to name different cats
contains a method to enable different cats to greet each other
"""

from cat import Cat

tom = Cat("tom")
melvin = Cat("melvin")

tom.greet(melvin)
melvin.greet(tom)
