class item:
    
    def __init__(self, name, price):
        self.name = name
        self.price = price

class order:
    
    def __init__(self, customer):
        self.customer = customer
        self.cost = 0
        self.items_ordered = []
    
    def add_to_order(self, item):
        self.items_ordered.append(item)
        self.cost += item.price
        #return self.cost
    
    def summarise_order(self):
        print(f'Customer Name: {self.customer.name}')
        print('You order list')
        for i2 in self.items_ordered:
            i2.display_details()
        if self.customer.discount is True:
            savings = 0
            for i in self.items_ordered:
                savings += i.discount
            self.cost = self.cost - savings
            print(f'Total cost is {self.cost} KRW and you got {savings} KRW discounts')
        else: print(f'Total cost is {self.cost} KRW')    
            
class customer:
    
     def __init__(self, name, discount=False):
        self.name = name
        self.discount = discount

class drink(item):
    
    def __init__(self, name, price, size):
        super().__init__(name, price)
        self.size = size

class tea(drink):
    
    def __init__(self, name, price, size, flavour):
        super().__init__(name, price, size)
        self.flavour = flavour
        self.discount = 500
        
    def display_details(self):
        print(f'{self.name} {self.price:.2f} KRW, {self.size}, {self.flavour}')
        
class mineral_water(drink):
    
    def __init__(self, name, price, size, is_carbonated=False):
        super().__init__(name, price, size)
        self.is_carbonated = is_carbonated
        self.discount = 0
        
    def display_details(self):
        print(f'{self.name} {self.price:.2f} KRW, {self.size}')
        if self.is_carbonated:
            print('Sparking water')
        else:
            print('Still water')

class cake(item):
    
    def __init__(self, name, price, slice_size, types, has_nuts=False):
        super().__init__(name, price)
        self.slice_size = slice_size
        self.types = types
        self.discount = 1000
        if has_nuts == False:
            self.has_nuts = "does not have any nuts"
        else: self.has_nuts = "has nuts"
        
    def display_details(self):
        print(f'This {self.name} cake {self.has_nuts}.')
        print(f'The price is {self.price} KRW and size is {self.slice_size} | {self.types}.')


class sandwich(item):
    
    def __init__(self, name, price, bread_type, filling):
        super().__init__(name, price)
        self.bread_type = bread_type
        self.filling = filling
        self.discount = 500    
    
    def display_details(self):
        print(f'This {self.name} sandwich is made by {self.bread_type}')
        print(f'The price is {self.price} KRW and filling is {self.filling}.')

def main():
    # Create two customers...
    cust1 = customer('Harry Palmer', False)
    cust2 = customer('Bill Preston', True)  # A loyal regular customer

    # Order some items...
    order1 = order(cust1)
    order1.add_to_order(tea('Black tea', 2500, 'large', 'Earl Gray'))
    order1.add_to_order(sandwich('Club special', 4000, 'brown', 'chicken'))

    order2 = order(cust2)
    order2.add_to_order(mineral_water('Evian', 1500, 'small', False))
    order2.add_to_order(sandwich('Simple sandwich', 2500, 'white', 'cheese'))
    order2.add_to_order(cake('Chocolate dream', 6500, 'medium', 'chocolate', True))

    # Summarise our orders...
    order1.summarise_order()
    print()
    order2.summarise_order()
    print()


if __name__ == "__main__":
    main()
