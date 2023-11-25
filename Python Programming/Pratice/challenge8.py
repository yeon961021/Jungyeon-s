class Student:
    
    def __init__(self, firstname, lastname, course, module_list):
        self.firstname = firstname
        self.lastname = lastname
        self.name = lastname +" "+firstname
        self.course = course
        self.modules = module_list
        
    def show_details(self):
        print(f"{self.lastname} {self.firstname}'s course is {self.course} and registered moduels: {self.modules} ")
        
    def change_course(self, new_course):
        self.course = new_course
        #return self.course

class Module:
    
    def __init__(self, name, code, tutor):
        self.name = name
        self.code = code
        self.tutor = tutor
        self.student_list = []
    def enrol_student(self, a):
        self.student_list.append(a)
        return self.student_list
    def show_all_enrolled_students(self):
        print(f"Enrooled on module: {self.name} ")
        print(self.student_list)

def main():
    A101 = Module("A101", "A101", "Dr.")
    E102 = Module("E102", "E102", "Dr.")
    M105 = Module("M105", "M105", "Dr.")

    a = Student('Barlow', 'Ken', 'English', ["A101", "E102"])
    b = Student('Baldwin', 'Mike', 'Business', ["A101"])
    c = Student('Legg', 'Harold', 'Medicine', ["E102"])
    A101.enrol_student(a.name)
    A101.enrol_student(b.name)
    E102.enrol_student(a.name)
    E102.enrol_student(c.name)
    a.show_details()
    b.show_details()
    c.show_details()
    A101.show_all_enrolled_students()
    E102.show_all_enrolled_students()
    a.change_course("Engineering")
    a.show_details()

if __name__ == "__main__":
    main()
