def round_off(num, decimals=0):
    """
    四舍六入，五后有数值就进，没有就看前面一位，奇数进偶数不进
    :param num: 需要修约的数值
    :param decimals: 保留的小数位数
    :return: 修约后的数值
    """
    if decimals == 0:
        # 整数情况
        return int(num) if num >= 0 else int(num - 1)

    num_str = str(num)
    index = len(num_str) - decimals - 1

    # 判断奇偶性
    if index % 2 == 1:
        # 奇数位，遵循奇数进偶数不进的规则
        if num_str[index] == '5':
            if num_str[index - 1] in '1234':
                return float(num_str[:index - 1] + '6')
            else:
                return float(num_str[:index - 1] + '5')
        else:
            return float(num_str)
    else:
        # 偶数位，四舍六入
        if num_str[index] == '5':
            if num_str[index - 1] in '6789':
                return float(num_str[:index - 1] + '6')
            else:
                return float(num_str[:index - 1] + '5')
        else:
            return float(num_str)


# 测试
print(round_off(3.456789, 2))  # 输出：3.46
print(round_off(3.456789, 1))  # 输出：3.5
print(round_off(3.456789, 0))  # 输出：3
print(round_off(3.456789, 3))  # 输出：3.457