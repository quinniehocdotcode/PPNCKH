import threading
import time

shared_data = "Dữ liệu ban đầu"  # Biến toàn cục (có thể dùng ở mọi luồng)

def thread_1():
    global shared_data
    time.sleep(1)  # Giả sử đang xử lý gì đó
    print(shared_data)
    shared_data = "Luồng 1 đã cập nhật dữ liệu!"  # Thay đổi giá trị của biến

def thread_2():
    time.sleep(2)  # Đợi thread_1 cập nhật
    print("Luồng 2 thấy:", shared_data)  # Lấy giá trị từ thread_1

t1 = threading.Thread(target=thread_1)
t2 = threading.Thread(target=thread_2)

t1.start()
t2.start()

t1.join()
t2.join()
