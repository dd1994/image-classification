import ctypes
from ctypes import wintypes

# Windows constants
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


def prevent_sleep():
    """防止 Windows 系统休眠"""
    try:
        # 加载 kernel32.dll
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        # 设置函数参数类型
        kernel32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
        kernel32.SetThreadExecutionState.restype = wintypes.DWORD

        # 设置系统状态：保持系统运行、显示器开启
        execution_state = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        kernel32.SetThreadExecutionState(execution_state)
        print("已设置防止系统休眠")
    except Exception as e:
        print(f"设置防止休眠失败: {e}")


def restore_sleep():
    """恢复 Windows 系统休眠设置"""
    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
        kernel32.SetThreadExecutionState.restype = wintypes.DWORD

        # 恢复正常状态
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print("已恢复系统休眠设置")
    except Exception as e:
        print(f"恢复休眠设置失败: {e}")
