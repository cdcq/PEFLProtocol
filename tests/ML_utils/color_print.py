class bcolors:
    HEADER = '\033[95m'     #紫色
    OKBLUE = '\033[94m'     #蓝色
    OKCYAN = '\033[96m'     #亮蓝色
    OKGREEN = '\033[92m'    #绿色
    WARNING = '\033[93m'    #黄色
    FAIL = '\033[91m'       #红色
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_train(input):
    print(bcolors.OKGREEN + "[Train]  " + bcolors.ENDC + input)

def print_poison_train(input):
    print(bcolors.FAIL + "[Poison] " + bcolors.ENDC + input)

def print_test(input):
    print(bcolors.OKBLUE + "[Test1]  " + bcolors.ENDC + input)

def print_poison_test(input):
    print(bcolors.WARNING + "[Test2]  " + bcolors.ENDC + input)

