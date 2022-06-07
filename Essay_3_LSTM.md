<h1 align="center"><b>LSTM</b></h1>
<h6 align="center"><b>Danh sách thành viên</b></h6>

| Họ và tên         | MSSV     |
| ----------------- | -------- |
| Văn Viết Hiếu Anh | 1952122  |
| Lê Văn Phước      | 19522054 |
| Văn Viết Nhật     | 19521958 |

## Lịch sử hình thành các kiến trúc RNN. Lý do tại sao cần các kiến trúc này? Nêu lý do vì sao cần các mô hình sau ( kiểu ví dụ từ mô hình RNN vì sao cần LSTM, mô hình RNN lúc đó tại sao ko tốt?)

## Tìm hiểu sâu về LSTM (giải thích kỹ các thành phần trong LSTM)

## Cài đặt thực nghiệm trên tensorflow
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
# Lưu ma trận ids để sử dụng trong tương lai.
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
# Kiểm tra thử trên 10 batches
iterations = 10
for i in range(iterations):
    nextBatch, nextBatchLabels = getTestBatch()
    # Tính độ chính xác 'accuracy' trên các test batch và gán vào 'test_acc'
    test_acc = (sess.run(accuracy, {inputs: nextBatch, labels: nextBatchLabels})) * 100
    print("Accuracy for this batch:", test_acc)
```

## Kết luận
