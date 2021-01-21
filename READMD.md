# Seam Carving

Optimizing Seam Carving using parallel programming in Cuda

## Description

- Input: Ảnh RGB

- Output: Ảnh được thay đổi kích thước (theo chiều rộng) mà không làm biến dạng các đối tượng quan trọng

- Ý nghĩa thực tế của ứng dụng: 
    
    + Các phiên bản với các kích thước khác nhau của cùng một tấm ảnh để hiển thị trên các thiết bị khác nhau (máy tính, điện thoại, ...)

- Có cần tăng tốc: 

## Host Sequential 

### Design

- Chuyển ảnh từ RGB sang gray scale

- Chạy hàm kernel edge detection lên ảnh

- Tính energy map dựa vào kết quả của edge detection

- Tìm seam ít quan trọng nhất

- Bỏ seam đó ra và xuất ảnh

### Comments

## Device Parallel (Option 1)

### Analysis

- Song song hóa grayscale (mỗi thread 1 pixel)

- Song song hóa edge detection (mỗi thread 1 pixel)

- Song song tính energy map bằng cách tính từng dòng với double buffering (mỗi thread 1 cột)

### Design

### Comments

## Device Parallel (Option 2)

### Analysis

- Tối ưu hóa như (Option 1)

- Song song hóa phần tìm và bỏ seam

### Design

### Comments

## Tasks

- Boilerplate (input, output, timer, setup, ...)

- Edge Detection (Host)

- Edge Detection (Device)

- Find Least Significant Seam (Host)

- Find Least Significant Seam (Device)

    + One of two way

    + Two of two way

## References

- [Seam Carving | Week 2 | 18.S191 MIT Fall 2020 | Grant Sanderson](https://www.youtube.com/watch?v=rpB6zQNsbQU)

- [Seam Carving: Live Coding Session | Week 2 | MIT 18.S191 Fall 2020 | James Schloss](https://www.youtube.com/watch?v=ALcohd1q3dk)

- [Seam Carving - Wikipedia](https://en.wikipedia.org/wiki/Seam_carving)

- [Shai Avidan & Ariel Shamir, Seam Carving for Content-Aware Image Resizing](https://perso.crans.org/frenoy/matlab2012/seamcarving.pdf)
