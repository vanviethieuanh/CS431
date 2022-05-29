# VGG 

## Sơ lược về quá trình phát triển mạng thần kinh nhân tạo

### Giai đoạn sơ khởi

Ý tưởng ban đầu của mạng thần kinh nhận tạo cũng như nhiều phát minh nổi tiếng khác của con người, được xây dựng dựa trên nhưng khám phá từ thế giới tự nhiên, và hiển nhiên mạng lưới thần kinh nhân tạo bắt đầu như một mô hình về cách các tế bào thần kinh trong não hoạt động. 
Khởi điểm vào năm 1943, nhà sinh lý học thần kinh Warren McCulloch và nhà toán học Walter Pitts đã sử dụng các mạch điên đơn giản để mô phỏng các hành vi thông minh, ý tưởng được đẩy đi xa hơn vào năm 1949 The Organization of Behaviour đề xuất rằng các đường dẫn thần kinh được tăng cường qua mỗi lần được kích hoạt liên tiếp nhau, từ đó đến mô phỏng lại  các quá trình phức tạp của não.

Từ năm 1950, các nha khoa học bắt đầu đưa các mạng đơn giản này lên hệ thống máy tính, và cũng trong khoảng thời gian này với sự giúp sức của các nhà khoa học thần kinh, đã đề xuất ra ý tưởng về Perceptron vào 1958, để giải thích các quá trình quyết định phức tạp trong não bằng cách sử dụng một cổng ngưỡng tuyến tính, lấy tổng có trọng số và trả về ‘0’ nếu kết quả dưới ngưỡng và ‘1’ nếu không. Và hiểu nhiên nhược điểm lớn nhất của Perceptron, chỉ  có thể học cách tách các lớp có thể phân tách tuyến tính, làm cho đơn giản nhưng phi tuyến tính trở thành một rào cản lớn 

<p align="center">
  <img src="/Images/VGG/percertron.PNG" alt="percertron"/>
</p>

Mọi thứ chỉ bắt đầu có tiến triển vào năm 1959, Bernard Widrow và Marcian Hoff đã phát triển mạng thần kinh nhân tạo đầu tiên áp dụng thành công cho một vấn đề trong thế giới thực. Các hệ thống này được đặt tên là ADALINE và MADALINE  Đặc biệt ở điểm là những tế bào thần kinh nhân tạo này khác với các perceptron chúng trả về dưới dạng đầu ra, trong trường hợp này là đầu vào có trọng số. 
Nhưng nó củng chỉ là cải tiển nhỏ trông hàng tá rào cản này đến rào cản khác còn vướng mặt khi đó, những thành công ban đầu này đã làm tăng khả năng và tiềm năng của mạng thần kinh, mọi thứ vần dần tiền triển chậm chạp trong giai đoạn này, hày còn được gọi là "mua đông AI" khi tiền đầu tư cạn kiệt, sự quan tâm của xã hội nguội lạnh và nghiên cứu AI gần như không có tiến triển lớn.

Sự tan băng của mùa đông kéo dài hàng thập kỷ này bắt đầu vào năm 1982 tại Học viện Khoa học Quốc gia khi Jon Hopfield trình bày bài báo của mình Hopfield Net và trong khi cùng năm tại hội nghị Mỹ-Nhật, Nhật Bản đã công bố dự định bắt đầu nỗ lực thế hệ tiếp theo trên Mạng thần kinh. Điều này khiến nguồn tài chính bắt đầu chảy trở lại từ sự đầu từ của các quốc gia lo sợ bị bỏ lại phía sau. Không lâu sau, Viện Vật lý Hoa Kỳ, vào năm 1985 đã thành lập cuộc họp thường niên “Neural Networks in Computing”, sau đó là Hội nghị quốc tế đầu tiên về Neural Networks by the Institute of Electrical and Electronic Engineers (IEEE) tổ chức vào năm 1987.

<p align="center">
  <img src="/Images/VGG/IEEE.PNG" alt="IEEE"/>
</p>

Tuy nhiên, một khám phá khác đã giúp mạng lưới thần kinh thoát ra sự bế tắc. Backpropagation, một phương pháp được các nhà nghiên cứu nghĩ ra từ những năm 60 và liên tục được phát triển cho đến mùa đông AI, Backpropagation cùng với Gradient Descent tạo thành xương sống và sức mạnh của mạng thân kinh nhân tạo. Trong khi Gradient Descent liên tục cập nhật và di chuyển các trọng số về phía mức tối thiểu của hàm mất mát, thì sự lan truyền ngược đánh giá gradient chi phí mất mát w.r.t. trọng số, độ lớn và hướng của chúng được sử dụng bởi gradient để đánh giá kích thước và hướng của các hiệu chỉnh đối với các thông số trọng số.

<p align="center">
  <img src="/Images/VGG/Gradient_descend.png" alt="Gradient_descend"/>
</p>

Và do đó, vào những năm 1990, mạng Neural  đã quay trở lại, lần này thực sự bắt kịp trí tưởng tượng của thế giới và cuối cùng đã ngang bằng với, nếu không muốn nói là vượt qua kỳ vọng của nó. 

### Giai đoạn nhảy vọt 

Trước hết vào năm 1980, một kiến trúc giới thiếu các khái niệm mới về trích xuất đặc trưng(feature extraction) bao gồm các lớp pooling và các lớp convolution, đầy cũng chính lác các phần thần chính của mạng mạng nhân tạo tính chập(CNN). Do đó Neocognitron  là kiến trúc đầu tiền của loại lớp này hay là tiền thân sớm nhất của mạng nhân tạo tính chấp. Điều đặc biệt của mạng kiến trúc này là nó cũng dựa trên cảm hứng về cấu tạo tự nhiên của cấu trúc phần kinh thị giác ở các loại động vật cấp cao. Cấu trúc của mạng xây dựng luân phiên từ các lớp của tê bào C và tế bào S,các tế bào là các phép toán học. Các “ô S”nằm ở lớp đước kết nối với các i “ô C” nằm ở lớp tiếp theo của mô hình. Ý tưởng tổng thể là các đặc trưng cục bộ được trích xuất trong các giai đoạn thấp hơn dần dần được tích hợp vào các đặc trưng toàn cầu hơn. Nó được sử dụng để nhận dạng ký tự viết tay (tiếng Nhật) và các nhiệm vụ nhận dạng mẫu khác.

Dựa trên nền tảng kiến trúc Neocognitron, những kiến trúc phức tạp khác dần được ra đời với mục đích cải thiện và hoàn thiện trong việc sử lý ảnh. Mãi cho đến năm 1989, tên gọi về mạng thân kinh nhận tạo tính chấp mới thực sự bắt đầu dựa trên mạng kiến trúc LeNet-5 bởi Yann LeCun và các cộng sự. Nó được nâng cấp và phát triển tới năm 1998 với mục đích chính lúc đó là sử dụng vào việc phân loại chữ số viết tay và huấn luyện nó trên bộ dữ liệu MNIST. Tuy nhiên vào thời điểm đó do chưa có sự phát triển của dữ liệu và khả năng tính toán nên mạng CNN vẫn chưa có cơ hội bùng nổ, mãi đến năm 2009, Bộ dữ liệu ImageNet được giới thiệu và tạo ra sự thay đổi trong giới nghiên cứu. Đây là bộ dữ liệu lớn nhất so với các bộ dữ liệu từng có từ trước đến thời điểm đó. Với kích thước lên tới 1 triệu ảnh và phân bố đều trên 1000 nhãn.

<p align="center">
  <img src="/Images/VGG/imagenet-logo.png" alt="ImageNet-logo"/>
</p>

Từ năm 2010, dự án ImageNet thực hiện một cuộc thi phần mềm hàng năm, Thử thách nhận diện trực quan quy mô lớn của ImageNet (ImageNet Large Scale Visual Recognition Challenge – ILSVRC), nơi các thuật toán cạnh tranh để phân loại và phát hiện các đối tượng và cảnh vật một cách chính xác. Từ đó các kiến trục mạng CNNs mới đều đặc đánh giá độ hiệu quả và tín nhiệm dựa trên bộ dữ liệu ImageNet làm phép đo tiêu chuẩn. Hiệu suất được đo bằng test error rate. Năm 2010, test error rate là 28,2%, năm tiêp theo, các nhà nghiên cứu đã cải  từ 28,2% lên 25,8%

<p align="center">
  <img src="/Images/VGG/ImageNet.png" alt="ImageNet"/>
</p>

Vào năm 2012, Alex Krizhevsky và Geoffrey Hinton đã đưa ra một kiến trúc CNN phổ biến cho đến ngày nay là AlexNet, giúp giảm lỗi từ 25,8% xuống 16,4%, đây là một cải tiến đáng kể vào thời điểm đó, năm tiếp theo kiến trúc ZFNet là kiến trúc tiếp theo đặt được state of the art, đây là bản nâng cấp của mạng AlexNet dựa trên điều chỉnh các siêu tham số của mạng.

<p align="center">
  <img src="/Images/VGG/AlexNet.png" alt="AlexNet"/>
</p>

Chuyển sang năm 2014, một trong những đóng góp quan trọng mà năm 2014 chứng kiến là sự ra đời của một kiến trúc mới được gọi là VGGNet. VGGNet, được phát minh bởi Visual Geometry Group (tại Đại học Oxford). Với suy nghĩ rằng bằng cách làm cho CNN sâu hơn, người ta có thể giải quyết vấn đề tốt hơn và nhận đượctest error rate thấp hơn trong việc phân loại ImageNet. Nhiều kiến trúc có độ sâu khác nhau đã được thử nghiệm, với việc bằng cách tăng chiều sâu, mạng thân kinh có thể mô hình hóa nhiều điểm phi tuyến tính hơn, giúp tăng và cải thiện các đặc trưng hơn cho việc huấn luyện

<p align="center">
  <img src="/Images/VGG/VGGnet.jpg" alt="VGGnet"/>
</p>

Trong thế giới của mạng thần kinh nhân tạo, mạng nhân tạo tính chập(CNN) là một phân nhân lớn trong đại gia đình, và cũng tùy thuộc vào nhiệm vụ và các ràng buộc tương ứng, ngày nay có rất nhiều loại kiến trúc khác nhau. Được sử dụng rộng rãi trong các lĩnh vực khác nhau, và đặc biệt được sử dụng phổ biến nhất trong phân tính ảnh. Kể từ khi được khám phá, kiến trúc mạng CNN đã trải qua các bước phát triển nhanh chóng và trong những năm gần đây đã đạt được kết quả mà trước đây là điều bất khả thi. 

## 2 Tìm hiểu về vgg.

Tiếp theo ta sẽ tìm hiểu sâu hơn về mô hình VGGNet, một kiến trúc mạng nơ-ron tích chập cổ điển (CNN). VGG được phát triển để tăng độ sâu của các CNN như vậy nhằm tăng hiệu suất của mô hình.

Trong những năm qua, học sâu đã mang lại thành công to lớn trong một loạt các nhiệm vụ về thị giác máy tính. Lĩnh vực máy học mới này đã và đang phát triển nhanh chóng. Hiệu suất hiện đại của học sâu so với các phương pháp học máy truyền thống cho phép các ứng dụng mới trong nhận dạng hình ảnh , thị giác máy tính , nhận dạng giọng nói, dịch máy, hình ảnh y tế, người máy và nhiều ứng dụng khác.

### Vậy VGG là gì?

VGG là viết tắt của Visual Geometry Group; nó là một kiến trúc Mạng nơ-ron hội tụ sâu tiêu chuẩn (CNN) với nhiều lớp. "Sâu" đề cập đến số lượng các lớp với VGG-16 hoặc VGG-19 bao gồm 16 và 19 lớp phức hợp.

Kiến trúc VGG là cơ sở của các mô hình nhận dạng đối tượng mang tính đột phá. Được phát triển như một mạng nơ-ron sâu, VGGNet cũng vượt qua các đường cơ sở về nhiều tác vụ và bộ dữ liệu ngoài ImageNet. Hơn nữa, bây giờ nó vẫn là một trong những kiến trúc nhận dạng hình ảnh phổ biến nhất.

![VGG-neural-network-architecture](/Images/VGG/vgg-neural-network-architecture.png)

##Kiến trúc mạng thần kinh VGG

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

