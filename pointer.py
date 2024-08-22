num1=11
num2=num1
print("Before num1 update")
print("num1 :", num1)  # 11
print("num2 :", num2)  # 11
print("\nnum1 points to:", id(num1))  # num1 points to: 4388955960
print("num2 points to:", id(num2))  # num2 points to: 4388955960

num1=22
print("After num1 update")
print("num1 :", num1)  # num1 : 22
print("num2 :", num2)  # num2 : 11

print("\nnum1 points to:", id(num1))
# num1 points to: 4388956312
print("num2 points to:", id(num2))
# num2 points to: 4388955960

dict1 = {'value': 11}
dict2 = dict1
print("Before value is update")
print("dict1 :", dict1)  # 11
print("dict2 :", dict2)  # 11
print("\ndict1 points to:", id(dict1))  # dict1 points to: 4388955960
print("dict2 points to:", id(dict2))  # dict2 points to: 4388955960

dict1['value']=22
print("\nAfter value is updated")
print("dict1 :", dict1)  # dict1 : 22
print("dict2 :", dict2)  # dict2 : 22

print("\ndict1 points to:", id(dict1))
# dict1 points to: 4388956312
print("dict2 points to:", id(dict2))
# dict2 points to: 4388956312

dict3 = {'value':33}
dict2=dict3
print("dict1 :", dict1)  # dict1 : 22
print("dict2 :", dict2)  # dict2 : 33
print("dict3 :", dict3)  # dict2 : 33
dict1=dict2
print("dict1 :", dict1)  # dict1 : 33

print("\ndict1 points to:", id(dict1))
# dict1 points to: 4388956312
print("dict2 points to:", id(dict2))
# dict2 points to: 4388956312
print("dict3 points to:", id(dict3))
# dict2 points to: 4388956312
