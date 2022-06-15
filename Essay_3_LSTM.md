<h1 align="center"><b>LSTM</b></h1>
<h6 align="center"><b>Danh sách thành viên</b></h6>

| Họ và tên         | MSSV     |
| ----------------- | -------- |
| Văn Viết Hiếu Anh | 1952122  |
| Lê Văn Phước      | 19522054 |
| Văn Viết Nhật     | 19521958 |

[toc]

## 1 Sơ lược quá trình phát triển của LSTM

Có rất nhiều vấn đề về mô hình hóa tuần tự trong cuộc sống hàng ngày: dịch máy, nhận dạng giọng nói, phân loại văn bản, phân tích trình tự DNA, v.v ... Hầu hết những vấn đề này thuộc về học có giám sát. Đầu vào hoặc đầu ra của các mô hình là tuần tự với kích thước thay đổi. Các mạng nơ-ron thông thường gặp khó khăn khi xử lý đầu vào và đầu ra với các kích thước khác nhau. Do đó, ta cần một mạng nơ-ron đáp ứng điều đó.

Khái niệm RNN được đưa ra vào năm 1986. Và kiến trúc LSTM nổi tiếng được phát minh vào năm 1997. Số lượng kiến trúc nổi tiếng của RNN ít hơn nhiều so với kiến trúc của CNN. Như câu nói nổi tiếng “Một bức tranh có giá trị ngàn lời nói” ngụ ý rằng hình ảnh có nhiều thông tin và không gian hơn để điều khiển, vì vậy không có gì ngạc nhiên khi lịch sử tiến hóa của RNN không đầy màu sắc.

### 1.1 Kiến trúc RNN

#### 1.1.1 Dữ liệu dạng sequence

Khi xử lý dữ liệu dạng chuỗi mà chuỗi này có sự tương quan với nhau theo thứ tự như câu từ, âm thanh, chuỗi hình ảnh hay một bài toán ví dụ như biến động giá cổ phiếu, ta biết rằng những dữ liệu dạng này có liên kết với nhau.

Có một điểm nữa mà ta cần lưu ý đó ra con người ta không suy nghĩ ngắt quảng mỗi khi ta tiếp nhận những thông tin dữ liệu này. Ta thừa hưởng và lưu lại những thông tin trước đó để tiếp tục tiếp nhận thông tin mới. Ví như ta đi xem phim *Em và Trịnh* thì không chỉ xem mỗi đoạn giữa mà hiểu là tại sao Trịnh Công Sơn gặp và yêu Dao Ánh mà phải xem từ lúc Trịnh Công Sơn đi gặp Bích Diễm mới hiểu được.

Vì lẽ đó những mô hình neural network truyền thống thì không thể thỏa mãn yêu cầu này, đó có thể coi là một khuyết điểm chính của mạng nơ-ron truyền thống.

Mạng nơ-ron hồi quy (Recurrent Neural Network) sinh ra để giải quyết vấn đề đó. Mạng này chứa các vòng lặp bên trong cho phép thông tin có thể lưu lại được.

#### 1.1.2 RNN

Tất cả bắt đầu dựa trên Hopfield networks - một loại RNN đặc biệt \- được John Hopfield phát hiện (lại) vào năm 1982. Năm 1993, một hệ thống nén lịch sử thần kinh đã giải quyết một tác vụ "Very Deep Learning" yêu cầu hơn 1000 lớp tiếp theo trong một RNN được mở ra kịp thời.

Trong vài năm gần đây, việc ứng dụng RNN đã đưa ra được nhiều kết quả không thể tin nổi trong nhiều lĩnh vực: nhận dạng giọng nói, mô hình hóa ngôn ngữ, dịch máy, mô tả ảnh,… Danh sách vẫn còn đang được mở rộng tiếp.

Đằng sau sự thành công này chính là sự đóng góp của LSTM. LSTM là một dạng đặc biệt của mạng nơ-ron hồi quy, với nhiều bài toán thì nó tốt hơn mạng hồi quy thuần. Hầu hết các kết quả thú vị thu được từ mạng RNN là được sử dụng với LSTM. Trong bài viết này, ta sẽ cùng khám phá xem mạng LSTM là cái gì nhé.

### 1.2 LSTM

#### 1.2.1 Vấn đề phụ thuộc xa

Một điểm nổi bật của RNN chính là ý tưởng kết nối các thông tin phía trước để dự đoán cho hiện tại. Việc này tương tự như ta sử dụng các cảnh Trịnh Công Sơn nhìn Dao Ánh để hiểu được tại sao ông mong chờ Dao Ánh đến vậy. Nếu mà RNN có thể làm được việc đó thì chúng sẽ cực kì hữu dụng, tuy nhiên liệu chúng có thể làm được không? Câu trả lời là *còn tùy*.

Đôi lúc ta chỉ cần xem lại thông tin vừa có thôi là đủ để biết được tình huống hiện tại. Ví dụ xem một đoạn Dap Ánh khen tranh Trịnh Công Sơn vẽ đẹp thì cũng biết cô là người bọc tranh và Trịnh Công Sơn cảm kích điều đó. Trong tình huống này, khoảng cách tới thông tin có được cần để dự đoán là nhỏ, nên RNN hoàn toàn có thể học được.

Nhưng trong nhiều tình huống ta buộc phải sử dụng nhiều ngữ cảnh hơn để suy luận. Về mặt lý thuyết, rõ ràng là RNN có khả năng xử lý các phụ thuộc xa (long-term dependencies). Chúng ta có thể xem xét và cài đặt các tham số sao cho khéo là có thể giải quyết được vấn đề này. Tuy nhiên, đáng tiếc trong thực tế RNN có vẻ không thể học được các tham số đó. Vấn đề này đã được khám phá khá sâu bởi Hochreiter (1991) và Bengio, et al. (1994), trong các bài báo của mình, họ đã tìm được nhưng lý do căn bản để giải thích tại sao RNN không thể học được.

Tuy nhiên, rất cám ơn là LSTM không vấp phải vấn đề đó!

#### 1.2.2 LSTM

Mạng bộ nhớ dài-ngắn (Long Short Term Memory networks), thường được gọi là LSTM - là một dạng đặc biệt của RNN, nó có khả năng học được các phụ thuộc xa. LSTM được giới thiệu bởi Hochreiter & Schmidhuber (1997), và sau đó đã được cải tiến và phổ biến bởi rất nhiều người trong ngành. Chúng hoạt động cực kì hiệu quả trên nhiều bài toán khác nhau nên dần đã trở nên phổ biến như hiện nay.

LSTM được thiết kế để tránh được vấn đề phụ thuộc xa (long-term dependency). Việc nhớ thông tin trong suốt thời gian dài là đặc tính mặc định của chúng, chứ ta không cần phải huấn luyện nó để có thể nhớ được. Tức là ngay nội tại của nó đã có thể ghi nhớ được mà không cần bất kì can thiệp nào.



## 2. Tìm hiểu về mạng thần kinh nhân tạo Long short-term memory

Mạng Long short-term memory(LSTM) là một loại mạng thần kinh trong Recurrent neural network(RNN) có khả năng học trong các bài toán đầu vào là một trình tự hay một dạng chuỗi. Trước khi đi sâu vào chi tiết mạng LSTM, chúng ta sẽ giới thiệu qua về Recurrent neural network. Học sâu gồm 2 mô hình lớn chính là Convolutional Neural Network(CNN) được sử dụng cho các bài toán xử lý đầu vào là ảnh, tương tự với Recurrent neural network(RNN) được sử dụng cho bài toán đầu vào dử liệu dạng chuỗi(sequence)

### 2.1 Recurrent neural network 

#### 2.1.1 Mô hình Recurrent neural netword

Như chúng ta đã biết, khi con người suy nghĩ hay đưa ra một quyết định nào đó, không thể đưa ra kết quả hợp lý từ khi bắt đầu suy nghĩ hay quyết định ngay tại thời điểm đó. Hoặc có thể hình dung đơn giản hơn khi đọc từng những dòng này, khi ta chỉ chỉ chọn một chữ trong dòng để đọc, ta không thể hiểu được nghĩa của nó được sử dụng là gì trong câu. Điều này đơn giản là vì khi ta suy nghĩ học đọc, chúng ta hiểu mỗi chữ ở đây dựa vào từ bạn đã hiểu các chữ trước đó chứ không phải là đọc tới đâu bỏ hết hết đi tới đó, rồi lại bắt đầu suy nghĩ lại từ đầu tới chữ chúng ta đang đọc. Tuy nhiên với các mạng thân kình nhân tạo truyền thống hay riêng với mạng Convolutional Neural Network thì để làm được điều này là bất khả thi.

Do đó với sự ra đời của mạng Recurrent Neural Network còn một cách gọi khác là mạng thần kinh hồi quy, đã giải quyết được vấn đề đó. Với ý tưởng là mạng thần kinh lưu lại các thông tin bằng cách sử dụng các vòng lặp trong mạng.

<p align="center">
  <img src="/Images/LSTM/RNN-rolled.png"/>
</p>


Hình vẽ trên mô ta một đoạn của mạng thần kinh nhân tạo hồi quy <img src="https://render.githubusercontent.com/render/math?math=A"> với đầu vào là <img src="https://render.githubusercontent.com/render/math?math=x_t"> và đầu ra là <img src="https://render.githubusercontent.com/render/math?math=h_t">. Vòng lặp cho phép thông tin có thể truyền từ bước này qua bước khác trong cùng một mạng thần kinh.

<p align="center">
  <img src="/Images/LSTM/RNN-unrolled.png"/>
</p>


Có thể hình dung rằng một mạng nhân tạo thần kinh hồi quy là nhiều bản sao chép của một mạng nhân tạo thuần trong đó mỗi đầu ra của mạng này là đầu vào của một mạng sao chép khác


#### 2.1.2 Những vấn đề tồn tại trong mạng RNN

Như đã được đề cập ở phần 1, về mặt lý thuyết, rõ ràng là RNN có khả năng xử lý các phụ thuộc xa bằng cách cài đặt và xem xét các siêu tham số chính sác để giải quyết vấn đề này. Tuy nhiên trong thực tế, đáng tiếng là mạng RNN không thể học được một cách hiệu quả các phụ thuộc xa.
Ngoài ra gradient biến mất(Vanishing gradient problem) và gradient bùng nổ(Exploding gradient problem) là các vấn đề này sảy ra do việc lựa chọn hàm kích hoạt không hợp lý hoặc số lượng các lớp ẩn quá lớn. Đặc biệt cũng là những vấn đề thường gặp phải khi sử dụng các kỹ thuật tối ưu hóa trong huấn luyện mạng thần kinh hồi quy.

### 2.2 Long short term memory (LSTM)

Trong vài năm gần đây, dựa vào việc ứng dụng mạng RNN đã giúp giải quyết được nhiều bài toán ngoài sức tưỡng tượng trong nhiều lĩnh vực bao gồm: nhận dạng giọng nói, mô hình hóa ngôn ngữ, dịch máy, mô tả ảnh... .Đằng sau nhưng kết quả thành công tuyệt vời này là một phần không hề nhỏ sử đóng góp của mạng LSTM và cũng có thể gọi LSTM là một dạng đặc biệt của mạng thần kinh nhân tạo hồi quy.

#### 2.2.1 Ý tưởng xây dựng cốt lỏi của LSTM
Mạng bộ nhớ dài-ngắn (Long Short Term Memory networks), thường được gọi là LSTM, là một dạng đặc biệt của RNN, nhằm mục đích giải quyết các vần đề tốn tại của RNN, dó đó LSTM được thiết kế để tránh vấn đề phụ thuộc xa và có đặc tính là nhớ thông tin trong suốt thời gian dài.
Mọi mạng hồi quy đều có dạng là một chuỗi các mô-đun lặp đi lặp lại của mạng thần kinh nhân tạo. Với mạng RNN chuẩn, các mô-dun này có cấu trúc rất đơn giản, thường là một tầng <img src="https://render.githubusercontent.com/render/math?math=tanh">. LSTM kế thừa từ RNN và cũng có kiến trục mạng như vậy, nhưng các mô-đun trong nó có cấu trúc khác với mạng RNN chuẩn. Thay vì chỉ có một tầng mạng nơ-ron, chúng có tới 4 tầng tương tác với nhau một cách rất đặc biệt.
Chìa khóa của LSTM là trạng thái tế bào (cell state). Trạng thái tế bào là một dạng giống như băng truyền. Nó chạy xuyên suốt tất cả các mắt xích (các nút mạng) và chỉ tương tác tuyến tính đôi chút. Vì vậy mà các thông tin có thể dễ dàng truyền đi thông suốt mà không sợ bị thay đổi.

LSTM có khả năng bỏ đi hoặc thêm vào các thông tin cần thiết cho trạng thái tế báo, chúng được điều chỉnh cẩn thận bởi các nhóm được gọi là cổng (gate).

Các cổng là nơi sàng lọc thông tin đi qua nó, Với ví dụ ở đây cho thấy rằng chúng được kết hợp bởi một tầng mạng sigmoid và một phép nhân.

<p align="center">
  <img src="/Images/LSTM/LSTM3-gate.png"/>
</p>


Tầng sigmoid sẽ cho đầu ra là một số trong khoản <img src="https://render.githubusercontent.com/render/math?math=[0,1]"> , mô tả có bao nhiêu thông tin có thể được thông qua. Khi đầu ra là <img src="https://render.githubusercontent.com/render/math?math=0"> thì có nghĩa là không cho thông tin nào qua cả, còn khi là <img src="https://render.githubusercontent.com/render/math?math=1"> thì có nghĩa là cho tất cả các thông tin đi qua nó.

Mô hình LSTM gồm có 3 cổng như vậy để duy trì và điều hành trạng thái của tế bào.

#### 2.2.2 Cơ chế hoạt động chi tiết của LSTM

Bước đầu tiên của mô hình LSTM là quyết định xem thông tin nào cần được bỏ đi hay dữ lại trong trạng thái tế bào. Quyết định được đưa ra bời "tầng cổng quên"(forget gate layer).

<p align="center">
  <img src="/Images/LSTM/LSTM3-focus-f.png"/>
</p>

Nó sẽ lấy đầu vào là <img src="https://render.githubusercontent.com/render/math?math=h_{t-1}"> và <img src="https://render.githubusercontent.com/render/math?math=x_t"> rồi đưa ra kết quả là một số trong khoảng <img src="https://render.githubusercontent.com/render/math?math=[0,1]"> cho mỗi số trong trạng thái tế bào <img src="https://render.githubusercontent.com/render/math?math=C_{t-1}">. Với đẩu ra là <img src="https://render.githubusercontent.com/render/math?math=1"> thể hiện rằng nó giữ toàn bộ thông tin lại, ngược lại với <img src="https://render.githubusercontent.com/render/math?math=0"> thì toàn bộ thông tin sẽ bị loại bỏ đi.

<p align="center">
  <img src="/Images/LSTM/LSTM3-focus-i.png"/>
</p>

Ở bước tiếp theo là quyết định xem thông tin mới nào ta sẽ lưu vào trạng thái tế bào. Việc này gồm 2 phần. Đầu tiên là sử dụng một hàm <img src="https://render.githubusercontent.com/render/math?math=sigmoid"> được gọi là “tầng cổng vào” (input gate layer) để quyết định giá trị nào ta sẽ cập nhập. Tiếp theo là một tầng <img src="https://render.githubusercontent.com/render/math?math=tanh"> tạo ra một véc-tơ cho giá trị mới <img src="https://render.githubusercontent.com/render/math?math=C_t"> nhằm thêm vào cho trạng thái. Trong bước tiếp theo, ta sẽ kết hợp 2 giá trị đó lại để tạo ra một cập nhập cho trạng thái.
Tại bước kế tiếp này là cập nhật lại trạng thái của tế bào củ <img src="https://render.githubusercontent.com/render/math?math=C_{t-1}"> thành trạng thái tế bào mới <img src="https://render.githubusercontent.com/render/math?math=C_t">. Từ nhưng bước trước mô hình đã quyết định xem dữ lại bao nhiêu thông tin, lấy bao nhiêu thông tin, giờ chúng ta chỉ cần thực hiện là xong.

<p align="center">
  <img src="/Images/LSTM/LSTM3-focus-C.png"/>
</p>

Ta sẽ nhân trạng thái cũ với <img src="https://render.githubusercontent.com/render/math?math=f_{t}"> để bỏ đi những thông tin ta quyết định quên lúc trước. Sau đó cộng thêm <img src="https://render.githubusercontent.com/render/math?math=i_{t} * C_t">. Trạng thái mơi thu được này phụ thuộc vào việc ta quyết định cập nhập mỗi giá trị trạng thái ra sao

<p align="center">
  <img src="/Images/LSTM/LSTM3-focus-o.png"/>
</p>

Cuối cùng, ta cần quyết định xem ta muốn đầu ra là gì. Giá trị đầu ra sẽ dựa vào trạng thái tế bào, nhưng sẽ được tiếp tục sàng lọc. Đầu tiên, ta chạy một tầng chứa hàm <img src="https://render.githubusercontent.com/render/math?math=sigmoid"> để quyết định phần nào của trạng thái tế bào ta muốn xuất ra. Sau đó, ta đưa nó trạng thái tế bảo qua một hàm <img src="https://render.githubusercontent.com/render/math?math=tanh">  để co giá trị nó về khoảng<img src="https://render.githubusercontent.com/render/math?math=[-1,1]">, và nhân nó với đầu ra của cổng <img src="https://render.githubusercontent.com/render/math?math=sigmoid">  để được giá trị đầu ra ta mong muốn.


#### 2.2.3 Kết luận về mạng LSTM 

Nhưng bước được mô tả ở trên là một trong nhưng mô hình LSTM phổ biến nhất, dựa vào ý tưỡng và cách xây dựng đã được thao luận, chúng ta có thể tự xây dựng một mô hình LSTM phiên bản cho riêng bản thân với nhưng nâng cấp và yêu cầu tùy chỉnh phù hợp. Sự khác nhau có không lớn, nhưng chúng giúp giải quyết phần nào đó trong cấu trúc của LTSM.

<p align="center">
  <img src="/Images/LSTM/LSTM3-var-GRU.png"/>
</p>

Trên đây là một trong nhưng biên thể của mạng LSTM có tên là Gated Recurrent Unit, hay GRU. Nó kết hợp các cổng loại trừ và đầu vào thành một cổng “cổng cập nhập” (update gate). Nó cũng hợp trạng thái tế bào và trạng thái ẩn với nhau tạo ra một thay đổi khác. Kết quả là mô hình của ta sẽ đơn giản hơn mô hình LSTM chuẩn và ngày càng trở nên phổ biến.

Dựa trên nhưng gì đã được tìm hiểu ở trên, có thể nói rằng LSTM là một mạng cải tiến của RNN nhằm giải quyết vấn đề nhớ các bước dài của RNN. LSTM là một bước lớn trong việc sử dụng RNN. Ý tưởng của nó giúp cho tất cả các bước của RNN có thể truy vấn được thông tin từ một tập thông tin lớn hơn. Ví dụ, nếu bạn sử dụng RNN để tạo mô tả cho một bức ảnh, nó có thể lấy một phần ảnh để dự đoán mô tả từ tất cả các từ đầu vào. 

## 3. Cài đặt thực nghiệm trên tensorflow
Tương tự như các bài toán khác, khi ta hiểu lý thuyết rồi thì cũng phải tiến hành áp dụng thì mới hiểu sâu được. Vì thế, phần này sẽ tiến hành cài đặt kiến trúc LSTM với tensorflow để mang lại cách nhìn cụ thể từng bước thực hiện khi cài đặt mô hình này như thế nào?

Ta sẽ áp dụng kiến trúc LSTM này vào bài toán Phân tích cảm xúc (Sentiment Analysis) của người dùng trên tập dữ liệu văn bản. Nếu nhìn theo kiểu black box, đầu vào của bài toán là một câu hoặc đoạn văn bản và đầu ra là trạng thái tích cực, tiêu cực hay trung hoà (positive - negative - neutral). Trong bài toán này, chúng ta chỉ quan tâm đến hai trạng thái cảm xúc là positive và negative.

<p align="center">
  <img src="/Images/LSTM/input_output.png" alt="input_output"/>
</p>

Nếu như chúng ta giữ nguyên định dạng đầu vào là chuỗi ký tự thì rất khó để thực hiện các thao tác biến đổi như tích vô hướng (dot product) hoặc các thuật toán trên mạng neural network như backpropagation. Thay vì dữ liệu đầu vào là một chuỗi, chúng ta cần chuyển đổi các từ trong tập từ điển sang dạng vector số học trong đó có thể thực hiện được các phép toán nêu trên.

<p align="center">
  <img src="/Images/LSTM/word2vec.png" alt="input_output"/>
</p>

Trong hình minh hoạ ở trên, ta có thể hình dung dữ liệu đầu vào của thuật toán phân tích cảm xúc là một ma trận 16 x D chiều. Trong đó 16 là số lượng từ trong câu và D là số chiều của không gian vector để biểu diễn từ. Để ánh xạ từ một từ sang một vector, chúng ta sử dụng ma trận word embedding.

Trong phần cài đặt này, chúng tôi sử dụng tập dữ liệu review trên trang Foody với khoảng 30,000 mẫu được gán nhãn. Trong đó có 15,000 mẫu positive và 15,000 mẫu negative. Nguồn: https://streetcodevn.com/blog/dataset.

Để huấn luận trên mạng RNN thì ta sẽ có 5 bước chính để giải quyết bài toán phân loại cảm xúc trong văn bản:
1. Huấn luyện một mô hình phát sinh ra vector từ (như mô hình Word2Vec) hoặc tải lên các vector từ tiền huấn luyện.
2. Tạo ma trận ID cho tập dữ liệu huấn luyện
3. Tạo mô hình RNN với các đơn vị LSTM, sử dụng tensorflow
4. Huấn luyện mô hình RNN với dữ liệu ma trận đã tạo ở bước 2
5. Đánh giá mô hình đã huấn luyện với tập test

Đầu tiên, để có thể biến đổi một từ thành một vector, chúng ta sử dụng mô hình đã được huấn luyện trước đó (pretrained model). Mô hình đã train trước đó cho tiếng Việt được lấy ở đây: https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.vi.300.vec.gz

Tuy nhiên, số lượng từ vựng tiếng Việt được huấn luyện rất lớn, khoảng 2M từ. Mỗi từ được biểu diễn dưới dạng một vector 300 chiều. Với kích thước gốc của ma trận word embedding như vậy sẽ gây khó khăn cho việc load dữ liệu cũng như đưa vào thư viện tensorflow để huấn luyện nên chúng tôi đã tối giản lại với số lượng từ tối thiểu để có thể chạy được trên tập dữ liệu review về đồ ăn của Foody.

```python
import numpy as np
import os
# Gán vào đường dẫn thư mục hiện tại của bạn
currentDir = ''

wordsList = np.load(os.path.join(currentDir, 'wordsList.npy'))
print('Simplified vocabulary loaded!')
wordsList = wordsList.tolist()
#wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load(os.path.join(currentDir, 'wordVectors.npy'))
wordVectors = np.float32(wordVectors)
print ('Word embedding matrix loaded!')
```

Để chắc chắn mọi dữ liệu được load lên một cách chính xác, chúng ta cần kiểm tra xem số lượng từ trong từ điển rút gọn và số chiều của ma trận word embedding có khớp với nhau hay không? Trong trường hợp này số từ mà chúng tôi giữ lại là 19,899 và số chiều trong không gian biểu diễn là 300 chiều.

```python
print('Size of the vocabulary: ', len(wordsList))
print('Size of the word embedding matrix: ', wordVectors.shape)
```

Để có thể xác định được vector biểu diễn của một từ tiếng Việt. Đầu tiên chúng ta sẽ xác định xem vị trí của từ đó trong wordsList. Sau đó lấy vector ở dòng tương ứng trên trên ma trận wordVectors. 

```python
ngon_idx = wordsList.index('ngon')
print('Index of `ngon` in wordsList: ', ngon_idx)
ngon_vec = wordVectors[ngon_idx]
print('Vector representation of `ngon` is: ', ngon_vec)
```

Bước thứ hai, chúng ta sẽ khảo sát tập dữ liệu huấn luyện và tạo ma trận ID. Trong phần cài đặt này, chúng tôi sử dụng tập dữ liệu lấy từ trang web Foody trên miền dữ liệu liên quan đến ẩm thực. Tập dữ liệu bao gôm 15.000 review tích cực đặt trong thư mục 'positiveReviews' và 15.000 review tiêu cực đặt trong thư mục 'negativeReviews'. Do khối lượng dữ liệu lớn, nếu chúng ta chọn số lượng từ tối đa (maxSeqLength) quá cao thì sẽ bị lãng phí khi biểu diễn ở những câu review quá ngắn. Ngược lại, nếu sử dụng số lượng từ tối đa quá ít thì sẽ bị bỏ lỡ những từ quan trọng giúp cho việc phân tích cảm xúc.

```python
from os import listdir
from os.path import isfile, join
positiveFiles = ['positiveReviews/' + f for f in listdir('positiveReviews/') if isfile(join('positiveReviews/', f))]
negativeFiles = ['negativeReviews/' + f for f in listdir('negativeReviews/') if isfile(join('negativeReviews/', f))]
numWords = []
for pf in positiveFiles:
    with open(pf, "r", encoding='utf-8') as f:
        line=f.readline()
        counter = len(line.split())
        numWords.append(counter)       
print('Positive files finished')

for nf in negativeFiles:
    with open(nf, "r", encoding='utf-8') as f:
        line=f.readline()
        counter = len(line.split())
        numWords.append(counter)  
print('Negative files finished')

numFiles = len(numWords)
print('The total number of files is', numFiles)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords)/len(numWords))
```

Tiếp đến, chúng ta sẽ chuẩn hóa văn bản và tách từ. Để tiết kiệm công sức và cũng nằm ngoài phạm vi của khoá học, chúng tôi đã chuẩn bị sẵn tập dữ liệu đã được tách từ. Giữa hai từ có thể ghép lại để tạo thành một khái niệm mới chúng tôi sử dụng ký tự '_' để nối các từ đó. Ví dụ: 'sinh_viên', 'sinh_học'.

Chúng tôi chuẩn bị sẵn các hàm chuẩn hoá văn bản nhằm loại bỏ các ký tự đặc biệt. Tham khảo ở hàm 'cleanSentences'.

```python
# Loại bỏ dấu chấm câu, dấu ngoặc đơn, dấu chấm hỏi, v.v. và chỉ để lại các ký tự chữ và số.
import re
strip_special_chars = re.compile("[^\w0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())
```

Bây giờ chúng ta sẽ biểu diễn 30.000 review dưới dạng các chỉ số của các từ. Tập dữ liệu positive và negative sẽ được tính hợp lại thành một ma trận 30000x180. Trong đó 30000 là số lượng review và 180 là số lượng từ tối đa cho một câu. Do bước chuẩn bị này tốn khá nhiều tài nguyên tính toán nên sau khi tính toán xong, chúng ta sẽ lưu lại để sử dụng cho những lần chạy thí nghiệm sau. Ma trận lưu trữ các chỉ số này là: 'ids'.

Tiếp theo, chúng ta sẽ tiến hành tra cứu từng từ trong review, sau đó gán vào ma trận 'ids'. Trong đó chỉ số dòng của ma trận tương ứng với file review, chỉ số cột của ma trận tương ứng với một từ của review. Trường hợp từ nào không có trong tập từ điển thì ta sẽ gán bằng chỉ số của từ 'UNK' (unknow).

```python
ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
nFiles = 0
# Chỉ mục của từ không xác định
unk_idx = wordsList.index('UNK')

for pf in positiveFiles:
    with open(pf, "r", encoding="utf-8") as f:
        nIndexes = 0
        line=f.readline()
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()
        for word in split:
            if word not in wordsList:
                ids[nFiles][nIndexes] += unk_idx
            else:
                ids[nFiles][nIndexes] += wordsList.index(word)
            nIndexes = nIndexes + 1
            if nIndexes >= maxSeqLength:
                break
        nFiles = nFiles + 1 

print('Positive files are indexed!')
for nf in negativeFiles:
    with open(nf, "r", encoding="utf-8") as f:
        nIndexes = 0
        line=f.readline()
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()
        for word in split:
            if word not in wordsList:
                ids[nFiles][nIndexes] += unk_idx
            else:
                ids[nFiles][nIndexes] += wordsList.index(word)
            
            nIndexes = nIndexes + 1
            if nIndexes >= maxSeqLength:
                break
        nFiles = nFiles + 1 

print('Negative files are indexed!')
# Lưu ma trận ids để sử dụng trong tương lai
np.save(os.path.join(currentDir,'idsMatrix.npy'), ids)
```

Tiếp đến, chúng ta xây dựng hàm lấy dữ liệu train và test theo từng batch.

```python
from random import randint

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0): 
            # Pick positive samples randomly
            num = randint(1,13999)
            labels.append([1,0])
        else:
            # Pick negative samples randomly
            num = randint(15999,29999)
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(13999,15999)
        if (num <= 14999):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels
```

Bước thứ ba, xây dựng RNN Model với tensorflow. Đầu tiên chúng tôi sẽ khởi tạo các tham số cho mô hình mạng RNN với các cell là các LSTM. Kiến trúc mạng ở đây bao gồm 128 đơn vị cho mỗi lớp, số lượng layer là 2, số lượng phân lớp là 2 và số vòng lặp khi huấn luyện là 30000.

```python
# Khởi tạo tham số
numDimensions = 300
batchSize = 64
lstmUnits = 128
nLayers = 2
numClasses = 2
iterations = 30000
```

Để lưu trữ dữ liệu input và ouput, chúng ta sẽ sử dụng hai kiểu dữ liệu placeholder. Một trong những điều quan trọng nhất khi khởi tạo các biến input và output này là xác định kích thước của các tensor. Mỗi output của mạng (hay còn gọi là label) sẽ là một vector one hot với hai giá trị tương ứng với hai loại cảm xúc: [1, 0] cho positive và [0, 1] cho negative.

<p align="center">
  <img src="/Images/LSTM/data_batch.png" alt="data_batch"/>
</p>

Khởi tạo hai biến 'inputs' và 'labels' bằng kiểu placeholder.

```python
import tensorflow as tf
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
inputs = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
```

Sau đó tạo dữ liệu word vector từ khối dữ liệu đầu vào với ma trận word embedding. Nếu như quá trình khởi tạo đúng thì sẽ tạo ra các kiểu dữ liệu sau:

labels --> Tensor("Placeholder:0", shape=(64, 2), dtype=float32)

inputs --> Tensor("Placeholder_1:0", shape=(64, 10), dtype=int32)

<p align="center">
  <img src="/Images/LSTM/embedding_data.png" alt="embedding_data"/>
</p>

```python
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors, inputs)
```

Như vậy sau bước này chúng ta đã có dữ liệu để đưa vào mạng mạng các LSTM. Để khởi tạo một LSTM chúng ta sử dụng hàm tf.nn.rnn_cell.BasicLSTMCell. Hàm này cần tham số đầu vào là số lượng đơn vị muốn khởi tạo. Đây chính là một hyperparamter đã được khởi tạo trước đó.
Để chống lại việc overfitting, chúng ta sử dụng lớp dropout. 

Để tăng tính phức tạp cho kiến trúc mạng chúng ta chồng các lớp LSTM lên nhau (Stack LSTM Layers). Trong trường hợp này chúng ta sử dụng 2 lớp LSTM. Việc chồng thêm các lớp LSTM sẽ giúp cho mô hình có khả năng nhớ nhiều thông tin hơn nhưng đồng thời cũng làm tăng số lượng tham số khi huấn luyện. Điều này cũng có nghĩa là sẽ làm tăng thời gian huấn luyện cũng như là cần thêm nhiều dữ liệu hơn.

Cuối cùng là đưa toàn bộ dữ liệu đầu vào vào mạng các LSTM sử dụng hàm tf.nn.dynamic_rnn. Chi tiết kiến trúc mạng LSTM sử dụng cho bài tập này được mô tả trong hình sau:

<p align="center">
  <img src="/Images/LSTM/architecture.png" alt="architecture"/>
</p>

```python
def generate_a_lstm_layer():
    # Khởi tạo một LSTM layer với 'lstmUnits' unit sử dụng hàm tf.contrib.rnn.BasicLSTMCell

    # Sau đó tạo một lớp dropout để chống overfitting với hệ số out_keep_prob bằng 0.75
    # Sử dụng hàm tf.contrib.rnn.DropoutWrapper
    cell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.75)

    return cell

# Sau khi đã có hàm tạo một LSTM Layer, ta sử dụng hàm này để chồng các LSTM lên
# Stack các LSTM layer với hàm tf.nn.rnn_cell.MultiRNNCell
lm_cell = tf.nn.rnn_cell.MultiRNNCell([generate_a_lstm_layer() for _ in range(nLayers)], state_is_tuple=True)
# Feed data variable vào mạng LSTM sử dụng hàm tf.nn.dynamic_rnn
with tf.variable_scope('scope', reuse = tf.AUTO_REUSE ):
    outputs, states = tf.nn.dynamic_rnn(lm_cell, data, dtype=tf.float32)
print(outputs)
```

Sau khi ra khỏi mạng LSTM, biến outputs sẽ là một tensor có kích thước [batchSize x maxSeqLength x lstmUnits], cụ thể là [64 x 180 x 128].

Sau đó, chúng ta chỉ lấy dữ liệu ở LSTM cell cuối cùng và cho đi qua lớp kết nối đầy đủ để phân loại thành 2 trạng thái. Chỉ số của LSTM cell cuối cùng là 179 (do có 180 cell theo chiều ngang)  nên để có thể lấy được giá trị ta sẽ chuyển vị về tensor có kích thước [maxSeqLength x batchSize x lstmUnits] hay [180 x 64 x 128]. Sử dụng hàm tf.gather để lấy tensor thứ 179 có kích thước [64 x 128] bao gồm 64 mẫu vector 128 chiều. Vector 128 chiều này sẽ được đưa vào lớp fully connected để chuyển đổi về vector 2 chiều tương ứng với 2 trạng thái.

Lớp kết nối đầy đủ bao gồm các bộ tham số 'weight' và 'bias' để thực hiện việc dự đoán kết quả. Bước này chính là tạo một lớp Fully Connected như trong sơ đồ kiến trúc mạng LSTM.

```python
weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))

# Lấy giá trị output tại LSTM cell cuối cùng
outputs = tf.transpose(outputs, [1, 0, 2])
last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
# Đưa qua mạng Fully Connected mà không có activation function
prediction = (tf.matmul(last, weight) + bias)
```

Để xác định độ chính xác của hệ thống, ta đếm số lượng labels khớp với giá trị dự đoán (prediction). Sau đó tính độ chính xác bằng cách tính giá trị trung bình của các kết quả trả về đúng.

```python
correctResult = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctResult, tf.float32))
```

Sau đó chúng ta sẽ xác định hàm độ lỗi sử dụng softmax cross entropy được tính từ dữ liệu dự đoán và tập labels. Cuối cùng là chọn thuật toán tối ưu với tham số learning rate mặc định là 0.001. 

```python
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)
```

Bước thứ tư, huấn luyện. Với mỗi vòng lặp, ta sẽ lấy ra một batch dữ liệu train để đưa vào mạng sử dụng `feed_dict`. với các tham số input và label là các placeholders. Bước huấn luyện này được lặp lại cho đến khi hết số lần cần huấn luyện.

```python
import datetime

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

sess = tf.InteractiveSession()
writer = tf.summary.FileWriter(logdir, sess.graph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for i in range(iterations):
    # Lấy batch tiếp theo
    nextBatch, nextBatchLabels = getTrainBatch()
    # Tối ưu optimizer
    sess.run(optimizer, {inputs: nextBatch, labels: nextBatchLabels})
    
    # Viết summary bằng Tensorboard
    # if (i % 50 == 0):
    #     summary = sess.run(merged, {inputs: nextBatch, labels: nextBatchLabels})
    #     writer.add_summary(summary, i)

    # Save model every 2000 training iterations
    if (i % 2000 == 0 and i != 0):
        save_path = saver.save(sess, os.path.join(currentDir,"models/pretrained_lstm.ckpt"), global_step=i)
        print("saved to %s" % save_path)
writer.close()
```

Bước cuối cùng, load mô hình đã train và đánh giá mô hình. Thời gian huấn luyện mạng khá lâu, nên trong quá trình mạng đang được huấn luyện, ta sẽ lưu lại một số checkpoint. Để có thể test thử trên một checkpoint mới nhất ta sử dụng hàm tf.train.latest_checkpoint và truyền vào tên thư mục muốn lấy model mới nhất.

```python
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint(os.path.join(currentDir,'models')))
```

Sau đó, với mỗi batch dữ liệu test, ta sẽ tiến hành test và tính độ chính xác.

```python
# Kiểm tra thử trên 10 batch
iterations = 10
for i in range(iterations):
    nextBatch, nextBatchLabels = getTestBatch()
    # Tính độ chính xác 'accuracy' trên các test batch và gán vào 'test_acc'
    test_acc = (sess.run(accuracy, {inputs: nextBatch, labels: nextBatchLabels})) * 100
    print("Accuracy for this batch:", test_acc)
```

## Kết luận

Trong bài viết này, chúng tôi đã nói sơ qua về quá trình phát triển các mạng RNN cũng như trình bày sâu về LSTM và cài đặt nó với tensorflow để minh họa cách thực thi LSTM trong thực tế.

LSTM là kiến trúc RNN phổ biến nhất, kể cả sau hơn 20 năm ra đời. Lý do lớn nhất là Vanilla RNN không thể nhớ rõ quá khứ. Bộ tham số duy nhất trong Vanilla RNN phải xử lý và ghi nhớ quá nhiều thông tin và nó dễ bị quá tải.

Đối với thông tin tuần tự như cuộc hội thoại và văn bản, có một số thứ nguyên cần xử lý như nội dung cần nhấn mạnh, nội dung cần xuất và nội dung cần quên. LSTM giới thiệu ba loại bộ nhớ (cổng) được chọn lần lượt là Cổng vào, Cổng ra và Cổng Quên, cũng như chức năng Sigmoid để biểu thị tỷ lệ phần trăm thông tin cần xử lý. Việc có ba đơn vị ký ức hợp lý để xử lý các chiều khác nhau giúp cải thiện đáng kể khả năng của LSTM để ghi nhớ cả thông tin dài hạn và ngắn hạn.

