# Here we go
This is an implementation of different methods for text classification with tensorflow, all the methods and experiments can be found here [**文本分类实践**](http://note.youdao.com/noteshare?id=9d0a3e6c1769938c3b57399a6ec317ee&sub=7C38D325A9484DB19102054701C9B5EC)

Method|Accuracy|Time(CPU)|Time(GPU)|Summary
:-:|:-:|:-:|:-:|:-:
lstm|0.75|1h38min13s||最耗时，误差小，准确率高
cnn|0.74|46min38s||误差大，准确率低
cnn-3-max|0.72|46min34s||-
cnn-6-max|0.725|46min53s||-
c-lstm|0.76|59min54s||误差波动大，准确率波动大
fastText-tf|**0.79(调参后)**|13min28s|2min42s|时间短，误差小，准确率相对高
fastText-mask|0.77|13min31s||-
fastText-fb|0.77|2s|-|速度极快，准确率不低
