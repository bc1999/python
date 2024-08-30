class Node:
    def __init__(self, value):
        self.value=value
        self.next=None


class LinkedList:
    def __init__(self, value):
        new_node = Node(value)
        self.head = new_node
        self.tail = new_node
        self.length=1

    def get_head(self):
        print("Head: ", self.head.value)
        return self.head

    def get_tail(self):
        print("Tail: ", self.tail.value)
        return self.tail

    def get_length(self):
        print("Length: ", self.length)
        return self.length

    def print_list(self):
        temp = self.head
        while temp is not None:
            print(temp.value)
            temp = temp.next

    def append(self, value):
        new_node = Node(value)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self.length += 1
        return True

    def pop(self):
        if self.head is None: return None
        # if self.length ==0:
            # return None
        temp = self.head
        pre = self.head
        while temp.next:
            pre = temp
            temp = temp.next
        self.tail = pre
        self.tail.next = None
        self.length -= 1
        if self.length == 0:
            self.head = None
            self.tail = None
        return temp


    def prepend(self, value):
        new_node = Node(value)
        if self.head is None: # if self.length == 0:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head = new_node
        self.length += 1

        return True

    def pop_first(self):
        if self.head is None: return None
        temp = self.head
        self.head = self.head.next
        temp.next = None
        self.length -= 1

        if self.length == 0:
            self.tail=None
        return temp



    def insert(self, index, value):
        pass

    def remove(self, index):
        pass

print("my_linked_list = LinkedList(4): ")
my_linked_list = LinkedList(4)
print(my_linked_list.head.value)
# 4

print("my_linked_list2.append(2): ")
my_linked_list2 = LinkedList(1)
my_linked_list2.append(2)
my_linked_list2.print_list()
# 1
# 2

print("my_linked_list2.pop().value: ")
# (2) Items - Returns 2 Node
print(my_linked_list2.pop().value)

# (1) Items - Returns 1 Node
print(my_linked_list2.pop().value)

# (0) Items - Returns None
print(my_linked_list2.pop())

print("my_linked_list.get_head().value: ")
print(my_linked_list.get_head().value)

print("my_linked_list.get_tail().value: ")
print(my_linked_list.get_tail().value)

print("my_linked_list.get_length(): ")
print(my_linked_list.get_length())

print("my_linked_list.pop_first().value: ",my_linked_list.pop_first().value)
