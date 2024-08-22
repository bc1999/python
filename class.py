class Cookie:
    def __init__(self, color):
        self.color = color

    def get_color(self):
        return self.color

    def set_color(self, color):
        self.color = color


cookie_one = Cookie('green')
cookie_two = Cookie('blue')
print("cookie_one is ", cookie_one.get_color())
# cookie_one is  green

print("cookie_two is ",cookie_two.get_color())
# cookie_two is  blue

cookie_one.set_color('yellow')

print("\ncookie_one is ", cookie_one.get_color())
# cookie_one is  yellow

print("cookie_two is ",cookie_two.get_color())
# cookie_two is  blue