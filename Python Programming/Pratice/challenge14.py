# A simple 2 class implementation for an online shopping application.

class Customer:
    """Represents a customer for an online shop."""

    def __init__(self, name, email):
        self.name = name
        self.email = email
        self.basket = []
        self.cost = 0.0

    def buy_item(self, item):
        self.basket.append(item)
        self.cost += item.price

    def remove_item(self, item):
        """Removes an item from the basket.
           Returns True is removed, False if not present."""
        if item in self.basket:
            self.cost -= item.price
            self.basket.remove(item)
            return True
        else:
            print('Not in the basket.')
            return False

    def remove_item_by_name(self, name):
        """Removes an item from the basket.
           Returns True is removed, False if not present."""
        for i in self.basket:
            if i.name == name:
                self.cost -= i.price
                self.basket.remove(i)
                return True
        # If we get here, the named item was not in the basket.
        return False

    def show_cost(self):
        """Returns the total cost of the basket."""
        print(f'Total cost is £{self.cost:.2f}')
        return self.cost

    def discount_basket(self, rate):
        """Make a discount of rate %.
           Returns True is rate is between 10% and 50% inclusive, False otherwise."""
        if 10.0 <= rate <= 50.0:  # rate >= 10.0 and rate <= 50.0:
            for i in self.basket:
                disc = i.price * rate / 100
                i.price -= disc
                self.cost -= disc
            return True
        else:
            print('Invalid rate.')
            return False

class Item:
    """Represents something we can buy in an online shop."""

    def __init__(self, name, price):
        self.name = name
        self.price = price

    def change_name(self, new_name):
        self.name = new_name
