"""
油价处理


import csv

# WTI原油数据路径
oil_wti = "data/oil/WTI.csv"
# London原油数据路径
oil_london = "data/oil/London.csv"

london_list = []  # 每月收盘价格列表
with open(oil_london, 'r', encoding='gb18030', errors='ignore') as f:  # 打开
    reader = csv.reader(f)   # 读取
    head_row = next(reader)  # 获取首行
    for row in reader:
        price = list(row)[1]
        london_list.insert(0, price)   # 第一个插入 倒序

wti_list = []  # 每月收盘价格列表
with open(oil_wti, 'r', encoding='gb18030', errors='ignore') as f:  # 打开
    reader = csv.reader(f)   # 读取
    head_row = next(reader)  # 获取首行
    for row in reader:
        price = list(row)[1]
        wti_list.insert(0, price)   # 第一个插入  倒序

with open("afterData/oil_price.txt", 'w') as f:
    f.write("Oil Price(2014/1-2018/11)"+"\n")
    try:
        for i in range(max(len(london_list), len(wti_list))):
            avg = (float(london_list[i])+float(wti_list[i]))//2  # 油价的均值
            f.write(str(avg)+'\n')
    except IndexError:
        print("WTI的数据和London的数据量不同!")
"""

# 砂石运输行情、煤炭、金属矿石和综合运价指数

import os

base_Name = "data/sandstoneAndcoal/['2016年2月长江干散货运价指数-长江运价指数-中华人民共和国交通运输部'].txt"

composite_price = []
coal_price = []
metal_price = []
sandstone_price = []
for x in range(2015, 2019):
    for y in range(1, 13):
        file_name = "data/sandstoneAndcoal/['%d年%d月长江干散货运价指数-" \
                    "长江运价指数-中华人民共和国交通运输部'].txt" % (x, y)
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                read = f.readlines()
                allStr = []
                for line in read:
                    oldLine = line.strip().split(" ")
                    line = []
                    for i in oldLine:
                        if i != "":
                            line.append(i)
                    allStr.extend(line)
                for index in range(len(allStr)):
                    if allStr[index] == "干散货综合运价指数":
                        print(allStr[index+3])
                        composite_price.append(allStr[index+3])
                    if allStr[index] == "煤炭运价指数":
                        print(allStr[index+3])
                        coal_price.append(allStr[index+3])
                    if allStr[index] == "金属矿石运价指数":
                        print(allStr[index+3])
                        metal_price.append(allStr[index+3])
                    if allStr[index] == "矿建材料运价指数":
                        print(allStr[index+3])
                        sandstone_price.append(allStr[index+3])
                print("###############################################3")


with open("afterData/composite_price.txt", 'w') as f:
    f.write("composite_price(2015/10-2018/9"+'\n')
    for i in composite_price:
        f.write(i+'\n')

with open("afterData/coal_price.txt", 'w') as f:
    f.write("coal_price(2015/10-2018/9"+'\n')
    for i in coal_price:
        f.write(i+'\n')

with open("afterData/metal_price.txt", 'w') as f:
    f.write("metal_price(2015/10-2018/9"+'\n')
    for i in metal_price:
        f.write(i+'\n')

with open("afterData/sandstone_price.txt", 'w') as f:
    f.write("sandstone_price(2015/10-2018/9"+'\n')
    for i in sandstone_price:
        f.write(i+'\n')

print(composite_price)
print(coal_price)
print(metal_price)
print(sandstone_price)







