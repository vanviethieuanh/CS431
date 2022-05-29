# Kể 1 câu chuyện theo sườn sau: problem - why - solution

## 1. Tìm hiểu vấn đề mà thuật toán này cần giải quyết (vì sao có CNN chẳng hạn?). Nêu tiền thân các mô hình đã có trước VGG. Tại sao các mô hình đó không hiệu quả? Giải pháp sử dụng cho các mô hình trước đó

## 2 Tìm hiểu về vgg.

Tiếp theo ta sẽ tìm hiểu sâu hơn về mô hình VGGNet, một kiến trúc mạng nơ-ron tích chập cổ điển (CNN). VGG được phát triển để tăng độ sâu của các CNN như vậy nhằm tăng hiệu suất của mô hình.

Trong những năm qua, học sâu đã mang lại thành công to lớn trong một loạt các nhiệm vụ về thị giác máy tính. Lĩnh vực máy học mới này đã và đang phát triển nhanh chóng. Hiệu suất hiện đại của học sâu so với các phương pháp học máy truyền thống cho phép các ứng dụng mới trong nhận dạng hình ảnh , thị giác máy tính , nhận dạng giọng nói, dịch máy, hình ảnh y tế, người máy và nhiều ứng dụng khác.

### Vậy VGG là gì?

VGG là viết tắt của Visual Geometry Group; nó là một kiến trúc Mạng nơ-ron hội tụ sâu tiêu chuẩn (CNN) với nhiều lớp. "Sâu" đề cập đến số lượng các lớp với VGG-16 hoặc VGG-19 bao gồm 16 và 19 lớp phức hợp.

Kiến trúc VGG là cơ sở của các mô hình nhận dạng đối tượng mang tính đột phá. Được phát triển như một mạng nơ-ron sâu, VGGNet cũng vượt qua các đường cơ sở về nhiều tác vụ và bộ dữ liệu ngoài ImageNet. Hơn nữa, bây giờ nó vẫn là một trong những kiến trúc nhận dạng hình ảnh phổ biến nhất.

<img src="/Images/VGG/vgg-neural-network-architecture.png" alt="Multi-class classification"/>

Kiến trúc mạng thần kinh VGG

### VGG16 là gì?

Mô hình VGG, hay VGGNet, hỗ trợ 16 lớp còn được gọi là VGG16, là một mô hình mạng nơ ron tích tụ do A. Zisserman và K. Simonyan từ Đại học Oxford đề xuất. Các nhà nghiên cứu này đã công bố mô hình của họ trong bài báo nghiên cứu có tiêu đề, “Mạng lưới hợp hiến rất sâu để nhận dạng hình ảnh quy mô lớn”.

Mô hình VGG16 đạt được độ chính xác gần như 92,7% trong bài kiểm tra top 5 trong ImageNet. ImageNet là một tập dữ liệu bao gồm hơn 14 triệu hình ảnh thuộc gần 1000 lớp. Hơn nữa, nó là một trong những mô hình phổ biến nhất được đệ trình cho ILSVRC-2014 . Nó thay thế các bộ lọc kích thước hạt nhân lớn bằng một số bộ lọc kích thước hạt nhân 3 × 3 lần lượt, do đó tạo ra những cải tiến đáng kể so với AlexNet. Mô hình VGG16 đã được đào tạo bằng cách sử dụng GPU Nvidia Titan Black trong nhiều tuần.

Như đã đề cập ở trên, VGGNet-16 hỗ trợ 16 lớp và có thể phân loại hình ảnh thành 1000 loại đối tượng, bao gồm bàn phím, động vật, bút chì, chuột, v.v. Ngoài ra, mô hình này có kích thước đầu vào hình ảnh là 224 x 224.

## 3. Cài đặt thực nghiệm trên code tensorflow

