# A simple 2 class implementation for an online shopping
# application - this time raise and catching exceptions.

# Tasks
# =====
# 1. Define an exception for ItemNotPresentError.
# 2. Define an exception for InvalidDiscountRateError.
# 3. Rewrire discount_basket to raise an InvalidDiscountRateError.

class ItemNotPresentError(Exception): 
    def __init__(self, message):
        print(message)

class InvalidDiscountRateError(Exception): 
    def __init__(self, message):
        print(message)              

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
           Raises ItemNotPresentError if not present."""
        try:
            if item not in self.basket:
                raise ItemNotPresentError('Not in the basket.')
            else:
                self.cost -= item.price
                self.basket.remove(item)
        except ItemNotPresentError:
            pass

    def remove_item_by_name(self, name):
        """Removes an item from the basket.
           Raises ItemNotPresentError if not present."""
        try:
            for i in self.basket:
                if i.name == name:
                    self.cost -= i.price
                    self.basket.remove(i)
                    return
            # If we get here, the named item was not in the basket.
            raise ItemNotPresentError('Not in the basket.')
        except ItemNotPresentError:
            pass

    def show_cost(self):
        """Returns the total cost of the basket."""
        print(f'Total cost is £{self.cost:.2f}')
        return self.cost

    # Rewrite this code so that it raises an InvalidDiscountRateError
    # if the discount rate is not in the stated range...

    def discount_basket(self, rate):
        """Make a discount of rate %.
           Returns True is rate is between 10% and 50% inclusive, False otherwise."""
        try:
            if 10.0 <= rate <= 50.0:  # rate >= 10.0 and rate <= 50.0:
                for i in self.basket:
                    disc = i.price * rate / 100
                    i.price -= disc
                    self.cost -= disc
                    return True
            else:
                raise InvalidDiscountRateError('Invalid rate.')
        except InvalidDiscountRateError:
            pass

class Item:
    """Represents something we can buy in an online shop."""

    def __init__(self, name, price):
        self.name = name
        self.price = price

    def change_name(self, new_name):
        self.name = new_name

def main():
    c1 = Customer('bob', 'bob@acme.com')
    c2 = Customer('alice', 'alice@acme.com')
    c3 = Customer('jimmy', 'jimmy@acme.com')
    i1 = Item('small table', 50.0)
    i2 = Item('table lamp', 30.0)
    i3 = Item('rug', 100.0)

    c1.buy_item(i1)
    c1.buy_item(i2)
    c2.buy_item(i1)
    c2.buy_item(i3)
    c3.buy_item(i3)

    # Try to remove items that are not present.
    c1.remove_item_by_name('pot')
    c2.remove_item(i2)
    c2.discount_basket(55)
    
if __name__ == '__main__':
    main()
