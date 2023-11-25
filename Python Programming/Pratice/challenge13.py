# A simple contacts class

class ContactsBook:

    def __init__(self):
        self.contacts = self.setup_contacts()

    def setup_contacts(self):
        """The key is a person's name. The value in each case is
           a 3-tuple with (email, position, extension)."""
        return {'jane': ('jane@acme.com', 'manager', 1546),
             'rod': ('rod@acme.com', 'programmer', 8724),
             'freddy': ('freddy@acme.com', 'support', 8524)}

    def list_all_contacts(self):
        """Iterate through the dictionary to show all contacts."""
        for k, v in self.contacts.items():
            email, position, extension = v
            print(f'{k}: email: {email}, position: {position}, ext: {extension}')

    def get_number_of_contacts(self):
        """Returns the number of contacts in this book."""
        return len(self.contacts.keys())

    def add_new_contact(self, name, email, position, extension):
        """Add a new key/value pairing to the dictionary.
           Returns True if this is a new contact and can be added
           False otherwise."""
        if name not in self.contacts.keys():
            self.contacts[name] = (email, position, extension)
            return True
        else:
            return False

    def update_email(self, name, new_email):
        """Updates an email address for the named person.
           Returns True if it can be updated, False otherwise."""
        if name in self.contacts.keys():
            v = self.contacts[name]
            email, position, extension = v
            self.contacts[name] = (new_email, position, extension)
            return True
        else:
            return False

    def get_email(self, name):
        """Fetches an email address for a named person.
        Returns None if the named person is not in the contacts list."""
        if name in self.contacts:
            v = self.contacts[name]
            email, position, extension = v
            return email
        else:
            return None

    def search_by_name(self, name):
        """Search for person by name and display contact details."""
        if name not in self.contacts:
            print(f'Sorry, {name} not in the contacts list.')
            return False
        else:
            v = self.contacts[name]
            email, position, extension = v
            print(f'{name}: email: {email}, position: {position}, ext: {extension}')
            return True

    def print_all_keys(self):
        """Print all keys in the dictionary."""
        for k in self.contacts:
            print(k)
            

