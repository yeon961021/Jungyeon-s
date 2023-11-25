# A simple contacts application
#
# In this example, we introduce the concept of a global variable
# A global variable has scope over all of your program and can
# be accessed anywhere in your codebase. To create a global variable
# initialise it outside the scope of any function or class, or create it
# explicitly using the 'global' keyword.

def setup_contacts():
    """The key is a person's name. The value in each case is
       a 3-tuple with (email, position, extension)."""
    return \
        {
            'jane': ('jane@acme.com', 'manager', 1546),
            'rod': ('rod@acme.com', 'programmer', 8724),
            'freddy': ('freddy@acme.com', 'support', 8524)
         }

def list_all_contacts():
    """Iterate through the dictionary to show all contacts."""
    for i, i2 in contacts.items():
        print(f'{i}: {list(i2)}')

def add_new_contact(name, email, position, extension):
    """Add a new key/value pairing to the dictionary."""
    contacts[name] = (email, position, extension)

def search_by_name(name):
    """Search for person by name and display contact details."""
    if name in contacts.keys():
        a, b, c = contacts[name]
        print(f"({name}) Email: {a} | Position: {b} | Extension: {c}")
    else: print("Not in our contacts!")

def print_all_keys():
    """Print all keys in the dictionary."""
    list_keys = list(contacts.keys())
    for i in list_keys:
        print(i)


def main():
    global contacts  # Global variable (not recommended, but useful for this example)
    contacts = setup_contacts()

    list_all_contacts()

    add_new_contact('samira', 'samira@acme.com', 'legal', 6245)
    add_new_contact('john', 'john@acme.com', 'maintenance', 6134)
    list_all_contacts()

    search_by_name('freddy')
    search_by_name('samira')
    search_by_name('david')  # Not in our contacts list...

    # Print out all keys to confirm...
    print_all_keys()

if __name__ == "__main__":
    main()bal contacts  # Global variable (not recommended, but useful for this example)
    contacts = setup_contacts()

    list_all_contacts()

    add_new_contact('samira', 'samira@acme.com', 'legal', 6245)
    add_new_contact('john', 'john@acme.com', 'maintenance', 6134)
    list_all_contacts()

    search_by_name('freddy')
    search_by_name('samira')
    search_by_name('david')  # Not in our contacts list...

    # Print out all keys to confirm...
    print_all_keys()

if __name__ == "__main__":
    main()
