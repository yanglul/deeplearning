import time

import win32gui  # 界面模块
import win32process  # 进程模块
import win32api
import ctypes

kernel32 = ctypes.windll.LoadLibrary(r"kernel32.dll")  # 核心文件
PROCESS_ALL_ACCESS = (0x000F0000 | 0x00100000 | 0xFFF)  # 调用最高权限执行
window_handle = win32gui.FindWindow(None, "Sekiro")  # 找到窗口句柄
print("window_handle",window_handle)
pid2,process_id= win32process.GetWindowThreadProcessId(window_handle)  # 获取进程ID
print("process_id",process_id)
print("pid2",pid2)

process_handle = win32api.OpenProcess(PROCESS_ALL_ACCESS, False, process_id)  # 得到进程句柄
print("process_handle",process_handle)



def findAdress(phandle,basead,shift):
    tmpData = ctypes.c_void_p()
    kernel32.ReadProcessMemory(int(phandle), ctypes.c_void_p(basead), ctypes.byref(tmpData), 8, None)
    aimdata = ctypes.c_void_p()
    for s in shift[:-1]:
        kernel32.ReadProcessMemory(int(phandle), ctypes.c_void_p(tmpData.value+s), ctypes.byref(aimdata), 8, None)
        tmpData = aimdata
        print("tmp_data",aimdata)
        time.sleep(1)
    return aimdata.value+shift[-1]

aim = findAdress(process_handle,0x143D5AAC0,[0x68,0x238,0x7C])
print("金币地址：",aim)
data4 = ctypes.c_void_p()
kernel32.ReadProcessMemory(int(process_handle), ctypes.c_void_p(aim), ctypes.byref(data4),8, None)
print("金币数量",data4)



money_num = 99991
kernel32.WriteProcessMemory(int(process_handle),ctypes.c_void_p(aim), ctypes.byref(ctypes.c_long(int(money_num))), 8,None)



