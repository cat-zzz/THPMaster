"""
@project: THPMaster
@File   : test1.py
@Desc   :
@Author : gql
@Date   : 2024/7/14 16:45
"""
# 根据launch文件加载要导航的点,分别为x，y，yaw三个数值
goalListX = "1.42"
goalListY = "-1.23"
goalListYaw = "-90.559"
goals = [[float(x), float(y), float(yaw)] for (x, y, yaw) in
         zip(goalListX.split(","), goalListY.split(","), goalListYaw.split(","))]
print(goals)
# print(zip(goalListX.split(","), goalListY.split(",")))
tags = [1, 2, 3, 4, 5]
print(tags)
print(tags[0])
tags = tags[0:1]
print(tags)
if tags is None or tags is []:
    print('tags is none')
else:
    print('tags is not none')
if __name__ == '__main__':
    pass
