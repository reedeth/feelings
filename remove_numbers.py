sabotage_no_num = []

def no_numbers(s):
    for i in s:
        if not i.isdigit():
            sabotage_no_num.append(i)

result = ''.join(sabotage_no_num)

print(result, file=open("sabotage_no_num.txt", "a"))
