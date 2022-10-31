"""
Coding exercise 2.1 from group 22

contains a constructor to name different cats
contains a method to enable different cats to greet each other
"""

class Cat:
    """
    class of a cat
    """

    def __init__(self, name):
        """
        init function for class (constructor)
        also names the cat during initialization
        """
        self.name = name
        
    def greet(self, other):
        print("Hello I am {} ! I see you are also a cool fluffy kitty {}, let's together purr at the human, so that they shall give us food!".format(self.name, other.name))
