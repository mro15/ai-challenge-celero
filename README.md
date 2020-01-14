# AI challenge Celero
## Sentiment analysis in the Large Movie Review Dataset

### Solution:
The representation is extracted using Paragraph Vector. The classifier used is the Multi Layer Perceptron.

### Run:
For training the paragraph vector and the MLP classifier run:

```bash
$ python3 main.py --operation train --path <dataset path>
```
Example:
```bash
$ python3 main.py --operation train --path datasets/aclImdb/
```
To classify a single review run:

```bash
python3 main.py --operation execution --path <review file path> 
```
Example:
```bash
$ python3 main.py --operation execution --path datasets/aclImdb/train/pos/3502_9.txt
```
