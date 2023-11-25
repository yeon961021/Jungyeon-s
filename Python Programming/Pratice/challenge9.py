class Borrower:
    
    new_id_code = 1
    
    def __init__(self, firstname, lastname):
        self.firstname = firstname
        self.lastname = lastname
        self.name = firstname + " " + lastname
        self.id = Borrower.new_id_code
        Borrower.new_id_code += 1
        self.booksBorrowed = []
        
    def borrowBook(self, book):
        self.booksBorrowed.append(book.title)
        return self.booksBorrowed
    
    def showAllBooks(self):
        print(f"You have been borrowed {self.booksBorrowed}.")
    
    def show_BorrowerDetails(self):
        print(f"Name: {self.name}\nID: {self.id}")


class Book:
    
    def __init__(self, title, author, code):
        self.title = title
        self.author = author
        self.code = code
        self.onload = True
        
    def showtitle(self):
        print(f"This book title is {self.title}\nAuther is {self.author} and the book code is {self.code}.")


class Library:
    
    def __init__(self):
        self.allBorrowers = []
        self.allBooks = []
        
    def addbook(self, book):
        self.allBooks.append(book)
        return self.allBooks
    
    def addborrower(self, borrower):
        self.allBorrowers.append(borrower)
        return self.allBorrowers
    
    def lendbook(self, borrower, book):
        if book.onload is True:
            borrower.borrowBook(book)
            book.onload = False
            print(f"Lending book {book.title} to {borrower.name}")
        else:
            print("The book is already rented.")


def main():
    book1 = Book('Kafkas Motorbike', 'Bridget Jones', 1290)
    book2 = Book('Cooking with Custard', 'Jello Biafra', 1002)
    book3 = Book('History of Bagpipes', 'John Cairncross', 987)
    library = Library()
    library.addbook(book1)
    library.addbook(book2)
    library.addbook(book3)
    bor1 = Borrower('Kevin', 'Wilson')
    bor2 = Borrower('Rita', 'Shapiro')
    bor3 = Borrower('Max', 'Normal')
    library.addborrower(bor1)
    library.addborrower(bor2)
    library.addborrower(bor3)
    library.lendbook(bor1, book1)
    bor1.show_BorrowerDetails()
    bor1.showAllBooks()
    library.lendbook(bor2, book2)
    library.lendbook(bor2, book3)
    bor2.show_BorrowerDetails()
    bor2.showAllBooks()
    library.lendbook(bor3, book3)
    book3.showtitle()


if __name__ == "__main__":
    main()
