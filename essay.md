<h1 align="center"><b>Softmax Regression</b></h1>

## 1. GIỚI THIỆU
Hồi quy Softmax (Softmax Regression) là một thuật toán học có giám sát (supervised learning), mặc dù tên gọi có chứa từ "hồi quy" nhưng đây là thuật toán thuộc loại classification. Nó tính toán mối quan hệ giữa các đặc trưng trong input và output dựa trên hàm softmax. Thực tế cho thấy nó là một trong những thuật toán Machine Learning được sử dụng phổ biến nhất.

Hồi quy Softmax (hay hồi quy logistic đa thức) là tổng quát của hồi quy logistic trong trường hợp chúng ta muốn xử lý nhiều lớp. Trong hồi quy logistic, chung tôi giả định rằng các nhãn là nhị phân <img src="https://render.githubusercontent.com/render/math?math=y^{i} \in \{0, 1\}">, nhưng trong hồi quy Softmax cho phép chúng tôi xử lý <img src="https://render.githubusercontent.com/render/math?math=y^{i} \in \{1, ..., K\}"> với <img src="https://render.githubusercontent.com/render/math?math=K"> là số lớp.

Trong cài đặt hồi quy softmax, chúng tôi quan tâm đến phân loại nhiều lớp (thay vì chỉ phân loại nhị phân), và vì vậy nhãn y có thể đảm nhiệm <img src="https://render.githubusercontent.com/render/math?math=K"> các giá trị khác nhau, thay vì chỉ có hai. 

## 2. PHƯƠNG PHÁP
Ý tưởng của bài toán là tương tự như bài toán hồi quy logistic, bài toán hồi quy softmax thay thế hàm sigmoid thành hàm softmax để có thể sử dụng cho bài toán phân loại nhiều lớp hơn.

### 2.1 Hàm Softmax
Chúng ta cần một mô hình xác suất sao cho với mỗi input <img src="https://render.githubusercontent.com/render/math?math=x">, <img src="https://render.githubusercontent.com/render/math?math=a_i"> thể hiện xác suất để input đó rơi vào lớp i. Vậy điều kiện cần là các <img src="https://render.githubusercontent.com/render/math?math=a_i"> phải dương và tổng của chúng bằng 1. Để có thể thỏa mãn điều kiện này, chung ta cần nhìn vào mọi giá trị <img src="https://render.githubusercontent.com/render/math?math=z_i"> và dựa trên các quan hệ giữa các <img src="https://render.githubusercontent.com/render/math?math=z_i"> này để tính toán giá trị của <img src="https://render.githubusercontent.com/render/math?math=a_i">.
Ngoài các điều kiện <img src="https://render.githubusercontent.com/render/math?math=a_i"> lớn hơn 0 và có tổng bằng 1, chúng ta sẽ thêm một điều kiện cũng rất tự nhiên nữa, đó là: giá trị <img src="https://render.githubusercontent.com/render/math?math=z_i = \theta_i^T x"> càng lớn thì xác suất dữ liệu rơi vào lớp i càng cao.
Điều kiện cuối này chỉ ra rằng chúng ta cần một hàm đồng biến ở đây.

Chú ý rằng <img src="https://render.githubusercontent.com/render/math?math=z_i"> có thể nhận giá trị cả âm và dương. Vì thế ta sử dụng hàm <img src="https://render.githubusercontent.com/render/math?math=exp(z_i) = e^{z_i}"> thì có thể chắc chắn biến <img src="https://render.githubusercontent.com/render/math?math=z_i"> thành một số dương, đồng biến. Điều kiện cuối cùng, tổng các <img src="https://render.githubusercontent.com/render/math?math=a_i"> bằng 1 có thể được đảm bảo nếu:

<img src="https://render.githubusercontent.com/render/math?math=a_i = \frac{exp(z_i)}{\sum_{i=1}^C exp(z_j)},  \forall_i = 1, 2, ..., C"> 

Hàm số này, tính tất cả các <img src="https://render.githubusercontent.com/render/math?math=a_i"> dựa vào tất cả các <img src="https://render.githubusercontent.com/render/math?math=z_i">, thỏa mãn tất cả các điều kiện đã xét: dương, tổng bằng 1, giữ được thứ tự của <img src="https://render.githubusercontent.com/render/math?math=z_i">. Hàm số này được gọi là hàm softmax.

Lúc này, ta có thể giả sử rằng:

<img src="https://render.githubusercontent.com/render/math?math=P(y_k = i \mid x_k">; <img src="https://render.githubusercontent.com/render/math?math=\theta) = a_i">

Trong đó, <img src="https://render.githubusercontent.com/render/math?math=P(y_k = i \mid x_k">; <img src="https://render.githubusercontent.com/render/math?math=\theta)"> được hiểu là xác suất để một điểm dữ liệu <img src="https://render.githubusercontent.com/render/math?math=x"> rơi vào lớp thứ i nếu biết tham số mô hình (ma trận trọng số) là <img src="https://render.githubusercontent.com/render/math?math=\theta">.

### 2.2 Hàm mất mát và phương pháp tối ưu

#### 2.2.1 One hot coding
Với bài toán phân loại nhiều lớp thì mỗi output sẽ không còn là một giá trị tương ứng với mỗi lớp nữa mà sẽ là một vector có đúng một phần tử bằng 1, các phần tử còn lại bằng 0. Phần tử bằng 1 nằm ở vị trí tương ứng với lớp đó, thể hiện rằng điểm dữ liệu đang xét rơi vào lớp này với xác suất bằng 1. Cách mã hóa output này được gọi là one-hot coding. Khi sử dụng mô hình Softmax Regression, với mỗi đầu vào <img src="https://render.githubusercontent.com/render/math?math=x">, ta sẽ có đầu ra dự đoán là <img src="https://render.githubusercontent.com/render/math?math=a = softmax(W^{T}x)">.
Trong khi đó, đầu ra thực sự chúng ta có là vector <img src="https://render.githubusercontent.com/render/math?math=y"> được biểu diễn dưới dạng one-hot coding.

Hàm mất mát sẽ được xây dựng để tối thiểu sự khác nhau giữa đầu ra dự đoán <img src="https://render.githubusercontent.com/render/math?math=a"> và đầu ra thực sự <img src="https://render.githubusercontent.com/render/math?math=y">. Một lựa chọn đầu tiên ta có thể nghĩ tới là:

<img src="https://render.githubusercontent.com/render/math?math=L(\theta) = \sum_{i=1}^N \parallel a_i - y_i\parallel_2^2">

Tuy nhiên đây chưa phải là một lựa chọn tốt. Khi đánh giá sự khác nhau (hay khoảng cách) giữa hai phân bố xác suất (probability distributions), chúng ta có một đại lượng đo đếm khác hiệu quả hơn. Đại lượng đó có tên là cross entropy.

#### 2.2.2. Cross Entropy
Cross entropy giữa hai phân phối <img src="https://render.githubusercontent.com/render/math?math=p"> và <img src="https://render.githubusercontent.com/render/math?math=q"> được định nghĩa là:

<img src="https://render.githubusercontent.com/render/math?math=H(p, q) = E_p[-logq]">

Với <img src="https://render.githubusercontent.com/render/math?math=p"> và <img src="https://render.githubusercontent.com/render/math?math=q"> là rời rạc (như <img src="https://render.githubusercontent.com/render/math?math=p"> và <img src="https://render.githubusercontent.com/render/math?math=q"> trong bài toán của chúng ta), công thức này được viết dưới dạng:

<img src="https://render.githubusercontent.com/render/math?math=H(p, q) = - \sum_{i=1}^C p_i log q_i"> <img src="https://render.githubusercontent.com/render/math?math=(1)">

Để hiểu rõ hơn ưu điểm của hàm cross entropy và hàm bình phương khoảng cách thông thường, chúng ta cùng xem Hình 4 dưới đây. Đây là ví dụ trong trường hợp <img src="https://render.githubusercontent.com/render/math?math=C = 2"> và <img src="https://render.githubusercontent.com/render/math?math=p_1"> lần lượt nhận các giá trị <img src="https://render.githubusercontent.com/render/math?math=0.5">, <img src="https://render.githubusercontent.com/render/math?math=0.1"> và <img src="https://render.githubusercontent.com/render/math?math=0.8">.

<img src="/Images/crossentropy1.png" alt="Cross-Entropy" style="width:300px;"/> <img src="/Images/crossentropy2.png" alt="Cross-Entropy" style="width:300px;"/> <img src="/Images/crossentropy3.png" alt="Cross-Entropy" style="width:300px;"/>

Hình 4: So sánh giữa hàm cross entropy và hàm bình phương khoảng cách. Các điểm màu xanh lục thể hiện các giá trị nhỏ nhất của mỗi hàm.

Có hai nhận xét quan trọng sau đây:

  + Giá trị nhỏ nhất của cả hai hàm số đạt được khi <img src="https://render.githubusercontent.com/render/math?math=q = p"> tại hoành độ của các điểm màu xanh lục.

  + Quan trọng hơn, hàm cross entropy nhận giá trị rất cao (tức loss rất cao) khi <img src="https://render.githubusercontent.com/render/math?math=q"> ở xa <img src="https://render.githubusercontent.com/render/math?math=p">. Trong khi đó, sự chênh lệch giữa các loss ở gần hay xa nghiệm của hàm bình phương khoảng cách <img src="https://render.githubusercontent.com/render/math?math=(q - p)^2"> là không đáng kể. Về mặt tối ưu, hàm cross entropy sẽ cho nghiệm gần với <img src="https://render.githubusercontent.com/render/math?math=p"> hơn vì những nghiệm ở xa bị trừng phạt rất nặng.

Hai tính chất trên đây khiến cho cross entropy được sử dụng rộng rãi khi tính khoảng cách giữa hai phân phối xác suất.

Chú ý: Hàm cross entropy không có tính đối xứng <img src="https://render.githubusercontent.com/render/math?math=H(p,q) \neq H(q, p)">. Điều này có thể dễ dàng nhận ra ở việc các thành phần của <img src="https://render.githubusercontent.com/render/math?math=p"> trong công thức <img src="https://render.githubusercontent.com/render/math?math=(1)"> có thể nhận giá trị bằng 0, trong khi đó các thành phần của <img src="https://render.githubusercontent.com/render/math?math=q"> phải là dương vì <img src="https://render.githubusercontent.com/render/math?math=log(0)"> không xác định. Chính vì vậy, khi sử dụng cross entropy trong các bài toán supervised learning, <img src="https://render.githubusercontent.com/render/math?math=p"> thường là đầu ra thực sự vì đầu ra thực sự chỉ có 1 thành phần bằng 1, còn lại bằng 0 (one-hot), <img src="https://render.githubusercontent.com/render/math?math=q"> thường là đầu ra dự đoán, khi mà không có xác suất nào tuyệt đối bằng 1 hoặc tuyệt đối bằng 0 cả.

Với Softmax Regression, trong trường hợp có <img src="https://render.githubusercontent.com/render/math?math=C"> classes, loss giữa đầu ra dự đoán và đầu ra thực sự của một điểm dữ liệu <img src="https://render.githubusercontent.com/render/math?math=x_i"> được tính bằng:

<img src="https://render.githubusercontent.com/render/math?math=L(\theta \mid x_i,y_i) = - \sum_{j=1}^C y_{ji} log(a_{ji})">

Với <img src="https://render.githubusercontent.com/render/math?math=y_{ji}"> và <img src="https://render.githubusercontent.com/render/math?math=a_{ji}"> lần lượt là là phần tử thứ j của vector (xác suất) <img src="https://render.githubusercontent.com/render/math?math=y_i"> và <img src="https://render.githubusercontent.com/render/math?math=a_i">. Nhắc lại rằng đầu ra <img src="https://render.githubusercontent.com/render/math?math=a_i"> phụ thuộc vào đầu vào <img src="https://render.githubusercontent.com/render/math?math=x_i"> và ma trận trọng số <img src="https://render.githubusercontent.com/render/math?math=\theta">.

#### 2.2.3. Hàm mất mát cho Softmax Regression
Kết hợp tất cả các cặp dữ liệu <img src="https://render.githubusercontent.com/render/math?math=x_i, y_i, i = 1, 2, ..., N">, chúng ta sẽ có hàm mất mát cho Softmax Regression như sau:

<img src="https://render.githubusercontent.com/render/math?math=L(\theta \mid X, Y) = - \sum_{i=1}^N \sum_{j=1}^C y_{ji} log(\frac{\exp(\theta_j^T x_i)}{\sum_{k=1}^C \exp(\theta_k^T x_i)})">
 
Với ma trận trọng số <img src="https://render.githubusercontent.com/render/math?math=\theta"> là biến cần tối ưu. Hàm mất mát này trông có vẻ đáng sợ, nhưng đừng sợ, đọc tiếp các bạn sẽ thấy đạo hàm của nó rất đẹp (và đáng yêu).

#### 2.2.4. Tối ưu hàm mất mát
Với chỉ một cặp dữ liệu <img src="https://render.githubusercontent.com/render/math?math=(x_i, y_i">), ta có:

<img src="https://render.githubusercontent.com/render/math?math=L_i(\theta) \triangleq L(\theta \mid x_i, y_i) = - \sum_{j=1}^C y_{ji} log(\frac{\exp(\theta_j^T x_i}{\sum_{k=1}^C \exp(\theta_k^T x_i)}) = - \sum_{j=1}^C (y_{ji} \theta_j^T x_i - y_{ji} log(\sum_{k=1}^C \exp(exp(\theta_k^T x_i)) = - \sum_{j=1}^C y_{ji} \theta_j^T x_i \dotplus log(\sum_{k=1}^C \exp(\theta_k^T x_i))"> (3)

trong biến đổi ở dòng cuối cùng, tôi đã sử dụng quan sát: <img src="https://render.githubusercontent.com/render/math?math=\sum_{j=1}^C y_{ji} = 1"> vì nó là tổng các xác suất.

Tiếp theo ta sử dụng công thức:

<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial L_i (\theta)}{\partial \theta} = [\frac{\partial L_i (\theta)}{\partial \theta_1}, \frac{\partial L_i (\theta)}{\partial \theta_2}, ..., \frac{\partial L_i (\theta)}{\partial \theta_C}]"> (4)

Trong đó, gradient theo từng cột có thể tính được dựa theo (3):

<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial L_i (\theta)}{\partial \theta_j} = - y_{ji} x_i \dotplus \frac{\exp(\theta_j_T x_i)}{\sum_{k=1}^C \exp(\theta_k^T x_i)} x_i = - y_{ji} x_i \dotplus a_{ji} x_i = x_i (a_{ji} - y_{ji}) = e_{ji} x_i"> (5)

Trong đó, giá trị <img src="https://render.githubusercontent.com/render/math?math=e_{ji} = a_{ji} - y_{ji}"> có thể coi là sai số dự đoán.

Đến đây ta đã được biểu thức rất đẹp rồi. Kết hợp (4) và (5) ta có:

<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial L_i (\theta)}{\partial \theta} = x_i [e_{1i}, e_{2i}, ..., e_{Ci} ] = x_i e_j^T">

Từ đây ta cũng có thể suy ra rằng:

<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial L (\theta)}{\partial \theta} = \sum_{i=1}^N x_i e_i^T = X E^T">
với <img src="https://render.githubusercontent.com/render/math?math=E=A-Y">

Giả sử rằng chúng ta sử dụng SGD, công thức cập nhật cho ma trận trọng số <img src="https://render.githubusercontent.com/render/math?math=\theta"> sẽ là:

<img src="https://render.githubusercontent.com/render/math?math=\theta = \theta \dotplus \eta x_i (y_i - a_i)^T">

## THAM KHẢO
1. http://deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/
2. https://machinelearningcoban.com/2017/02/17/softmax/
3. https://pic.plover.com/MISC/symbols.pdf
