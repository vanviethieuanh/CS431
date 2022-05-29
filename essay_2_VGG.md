# Kể 1 câu chuyện theo sườn sau: problem - why - solution

## 1. Tìm hiểu vấn đề mà thuật toán này cần giải quyết (vì sao có CNN chẳng hạn?). Nêu tiền thân các mô hình đã có trước VGG. Tại sao các mô hình đó không hiệu quả? Giải pháp sử dụng cho các mô hình trước đó

## 2 Tìm hiểu về vgg.

Tiếp theo ta sẽ tìm hiểu sâu hơn về mô hình VGGNet, một kiến trúc mạng nơ-ron tích chập cổ điển (CNN). VGG được phát triển để tăng độ sâu của các CNN như vậy nhằm tăng hiệu suất của mô hình.

Trong những năm qua, học sâu đã mang lại thành công to lớn trong một loạt các nhiệm vụ về thị giác máy tính. Lĩnh vực máy học mới này đã và đang phát triển nhanh chóng. Hiệu suất hiện đại của học sâu so với các phương pháp học máy truyền thống cho phép các ứng dụng mới trong nhận dạng hình ảnh , thị giác máy tính , nhận dạng giọng nói, dịch máy, hình ảnh y tế, người máy và nhiều ứng dụng khác.

### Vậy VGG là gì?

VGG là viết tắt của Visual Geometry Group; nó là một kiến trúc Mạng nơ-ron hội tụ sâu tiêu chuẩn (CNN) với nhiều lớp. "Sâu" đề cập đến số lượng các lớp với VGG-16 hoặc VGG-19 bao gồm 16 và 19 lớp phức hợp.

Kiến trúc VGG là cơ sở của các mô hình nhận dạng đối tượng mang tính đột phá. Được phát triển như một mạng nơ-ron sâu, VGGNet cũng vượt qua các đường cơ sở về nhiều tác vụ và bộ dữ liệu ngoài ImageNet. Hơn nữa, bây giờ nó vẫn là một trong những kiến trúc nhận dạng hình ảnh phổ biến nhất.

<img src="/Images/VGG/vgg-neural-network-architecture.png" alt="VGG-neural-network-architecture"/>

Kiến trúc mạng thần kinh VGG

### VGG16 là gì?

Mô hình VGG, hay VGGNet, hỗ trợ 16 lớp còn được gọi là VGG16, là một mô hình mạng nơ ron tích tụ do A. Zisserman và K. Simonyan từ Đại học Oxford đề xuất. Các nhà nghiên cứu này đã công bố mô hình của họ trong bài báo nghiên cứu có tiêu đề, “Mạng lưới hợp hiến rất sâu để nhận dạng hình ảnh quy mô lớn”.

Mô hình VGG16 đạt được độ chính xác gần như 92,7% trong bài kiểm tra top 5 trong ImageNet. ImageNet là một tập dữ liệu bao gồm hơn 14 triệu hình ảnh thuộc gần 1000 lớp. Hơn nữa, nó là một trong những mô hình phổ biến nhất được đệ trình cho ILSVRC-2014 . Nó thay thế các bộ lọc kích thước hạt nhân lớn bằng một số bộ lọc kích thước hạt nhân 3 × 3 lần lượt, do đó tạo ra những cải tiến đáng kể so với AlexNet. Mô hình VGG16 đã được đào tạo bằng cách sử dụng GPU Nvidia Titan Black trong nhiều tuần.

Như đã đề cập ở trên, VGGNet-16 hỗ trợ 16 lớp và có thể phân loại hình ảnh thành 1000 loại đối tượng, bao gồm bàn phím, động vật, bút chì, chuột, v.v. Ngoài ra, mô hình này có kích thước đầu vào hình ảnh là 224 x 224.

<img src="/Images/VGG/vgg16-deep-learning-objects.png" alt="Example-for-VGG16"/>

Ứng dụng pháp hiện đối tượng thời gian thực

### VGG19 là gì?

Khái niệm về mô hình VGG19 (cũng là VGGNet-19) giống với VGG16 ngoại trừ việc nó hỗ trợ 19 lớp. “16” và “19” là đại diện cho số lớp trọng lượng trong mô hình (lớp chập). Điều này có nghĩa là VGG19 có nhiều lớp chập hơn VGG16. Chúng ta sẽ thảo luận thêm về các đặc điểm của mạng VGG16 và VGG19 trong phần sau của bài viết này.

### Kiến trúc VGG

VGGNets dựa trên các tính năng thiết yếu nhất của mạng nơ-ron tích tụ (CNN). Hình ảnh sau đây cho thấy khái niệm cơ bản về cách thức hoạt động của CNN:

<img src="/Images/VGG/how-vgg-works-convolutional-neural-network.png" alt="How-vgg-works-convolutional-neural-network"/>

Kiến trúc của mạng nơ-ron hợp hiến: Dữ liệu hình ảnh là đầu vào của CNN, đầu ra của mô hình cung các các lớp dự đoán cho hình ảnh đầu vào.

Mạng VGG được xây dựng với các bộ lọc chập rất nhỏ. VGG-16 bao gồm 13 lớp phức hợp và ba lớp được kết nối đầy đủ.

Hãy cùng tìm hiểu sơ lược về kiến trúc của VGG:

+ Đầu vào: VGGNet có kích thước đầu vào hình ảnh là 224 × 224. Đối với cuộc thi ImageNet, những người tạo ra mô hình đã cắt bỏ mảng trung tâm 224 × 224 trong mỗi hình ảnh để giữ cho kích thước đầu vào của hình ảnh nhất quán.
+ Các lớp liên kết: Các lớp tích tụ của VGG tận dụng một trường tiếp nhận tối thiểu, tức là 3 × 3, kích thước nhỏ nhất có thể mà vẫn chụp lên / xuống và trái / phải. Hơn nữa, cũng có các bộ lọc tích chập 1 × 1 hoạt động như một phép biến đổi tuyến tính của đầu vào. Tiếp theo là đơn vị ReLU, đây là một sự đổi mới rất lớn từ AlexNet giúp giảm thời gian đào tạo. ReLU là viết tắt của chức năng kích hoạt đơn vị tuyến tính được chỉnh lưu; nó là một hàm tuyến tính từng mảnh sẽ xuất ra đầu vào nếu dương; nếu không, đầu ra bằng không. Sải tích chập được cố định ở 1 pixel để giữ nguyên độ phân giải không gian sau khi tích chập (sải chân là số pixel dịch chuyển trên ma trận đầu vào).
+ Các lớp ẩn: Tất cả các lớp ẩn trong mạng VGG đều sử dụng ReLU. VGG thường không tận dụng Chuẩn hóa phản hồi cục bộ (LRN) vì nó làm tăng mức tiêu thụ bộ nhớ và thời gian đào tạo. Hơn nữa, nó không cải thiện độ chính xác tổng thể.
+ Các lớp được kết nối đầy đủ: VGGNet có ba lớp được kết nối đầy đủ. Trong số ba lớp, hai lớp đầu tiên có 4096 kênh mỗi lớp và lớp thứ ba có 1000 kênh, mỗi lớp 1 kênh.

<img src="/Images/VGG/fully-connected-layers.png" alt="Fully-connected-layers"/>

Các lớp được kết nối đầy đủ

### Kiến trúc VGG16

Số 16 trong tên VGG ám chỉ thực tế rằng nó là mạng nơ-ron sâu 16 lớp (VGGnet). Điều này có nghĩa là VGG16 là một mạng khá rộng và có tổng số khoảng 138 triệu tham số. Ngay cả theo các tiêu chuẩn hiện đại, nó là một mạng lưới khổng lồ. Tuy nhiên, sự đơn giản của kiến trúc VGGNet16 là điều làm cho mạng trở nên hấp dẫn hơn. Chỉ cần nhìn vào kiến trúc của nó, có thể nói rằng nó khá đồng đều.

Có một vài lớp tích chập theo sau là một lớp gộp làm giảm chiều cao và chiều rộng. Nếu chúng ta nhìn vào số lượng bộ lọc mà chúng ta có thể sử dụng, có khoảng 64 bộ lọc có sẵn, chúng ta có thể tăng gấp đôi lên khoảng 128 và sau đó là 256 bộ lọc. Trong các lớp cuối cùng, chúng ta có thể sử dụng 512 bộ lọc.

<img src="/Images/VGG/VGG-16-architecture-of-the-model.png" alt="VGG-16-architecture-of-the-model"/>

Kiến trúc VGG-16 của mô hình VGG16

### Sự phức tạp và thách thức

Số lượng bộ lọc mà chúng ta có thể sử dụng tăng gấp đôi trên mỗi bước hoặc qua mỗi ngăn xếp của lớp tích chập. Đây là một nguyên tắc chính được sử dụng để thiết kế kiến trúc của mạng VGG16. Một trong những nhược điểm quan trọng của mạng VGG16 là nó là một mạng khổng lồ, có nghĩa là cần nhiều thời gian hơn để đào tạo các tham số của nó.

Do độ sâu và số lượng các lớp được kết nối đầy đủ, mô hình VGG16 có dung lượng hơn 533MB. Điều này làm cho việc triển khai mạng VGG trở thành một công việc tốn nhiều thời gian.

Mô hình VGG16 được sử dụng trong một số vấn đề phân loại hình ảnh học sâu, nhưng các kiến trúc mạng nhỏ hơn như GoogLeNet và SqueezeNet thường được ưu tiên hơn. Trong mọi trường hợp, VGGNet là một khối xây dựng tuyệt vời cho mục đích học tập vì nó rất dễ thực hiện.

### Hiệu suất của mô hình VGG

VGG16 vượt trội hơn rất nhiều so với các phiên bản trước của các mô hình trong các cuộc thi ILSVRC-2012 và ILSVRC-2013. Hơn nữa, kết quả VGG16 đang cạnh tranh cho người chiến thắng nhiệm vụ phân loại (GoogLeNet với sai số 6,7%) và vượt trội hơn đáng kể so với Clarifai đã giành giải ILSVRC-2013. Nó thu được 11,2% với dữ liệu đào tạo bên ngoài và khoảng 11,7% khi không có nó. Về hiệu suất một mạng, mẫu VGGNet-16 đạt kết quả tốt nhất với sai số thử nghiệm khoảng 7,0%, qua đó vượt qua một GoogLeNet khoảng 0,9%.

### VGGNet so với ResNet

VGG là viết tắt của Visual Geometry Group và bao gồm các khối, trong đó mỗi khối bao gồm các lớp 2D Convolution và Max Pooling. Nó có hai mẫu - VGG16 và VGG19 - với 16 và 19 lớp.

Khi số lượng lớp trong CNN tăng lên, khả năng mô hình phù hợp với các chức năng phức tạp hơn cũng tăng lên. Do đó, nhiều lớp hơn hứa hẹn hiệu suất tốt hơn. Không nên nhầm lẫn điều này với Mạng thần kinh nhân tạo (ANN), trong đó việc tăng số lượng lớp không nhất thiết dẫn đến hiệu suất tốt hơn.

Bây giờ câu hỏi là, tại sao bạn không nên sử dụng VGGNet với nhiều lớp hơn, chẳng hạn như VGG20 hoặc VGG50 hoặc VGG100? Đây là nơi mà vấn đề phát sinh. Trọng số của mạng nơ-ron được cập nhật thông qua thuật toán lan truyền ngược, thực hiện một thay đổi nhỏ đối với mỗi trọng số để giảm tổn thất của mô hình.

Nhưng nó xảy ra như thế nào? Nó cập nhật từng trọng lượng để nó thực hiện một bước theo hướng mà sự mất mát giảm xuống. Đây không là gì ngoài gradient của trọng lượng này có thể được tìm thấy bằng cách sử dụng quy tắc chuỗi.

Tuy nhiên, khi gradient tiếp tục chảy ngược trở lại các lớp ban đầu, giá trị sẽ tiếp tục tăng theo từng gradient cục bộ. Điều này dẫn đến gradient ngày càng nhỏ hơn, do đó làm cho các thay đổi đối với các lớp ban đầu rất nhỏ. Điều này làm tăng thời gian đào tạo đáng kể.

Vấn đề có thể được giải quyết nếu gradient cục bộ trở thành 1. Đây là lúc ResNet xuất hiện trong bức tranh vì nó đạt được điều này thông qua chức năng nhận dạng. Vì vậy, khi gradient được lan truyền ngược, nó không giảm giá trị vì gradient cục bộ là 1.

Các mạng phần dư sâu (ResNets), chẳng hạn như mô hình ResNet-50 phổ biến, là một loại kiến trúc mạng nơ-ron phức hợp (CNN) có chiều sâu 50 lớp. Mạng nơ-ron dư sử dụng việc chèn các kết nối tắt để biến một mạng thuần túy thành đối tác mạng còn lại của nó. So với VGGNets, ResNets ít phức tạp hơn vì chúng có ít bộ lọc hơn.

ResNet, còn được gọi là Mạng dư, không cho phép sự cố gradient biến mất xảy ra. Các kết nối bỏ qua hoạt động như siêu xa lộ gradient, cho phép gradient chảy không bị xáo trộn. Đây cũng là một trong những lý do quan trọng nhất tại sao ResNet có các phiên bản như ResNet50, ResNet101 và ResNet152.


## 3. Cài đặt thực nghiệm trên code tensorflow

