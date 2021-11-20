# Homework2: Tensorflow Flower Dataset
This is homework2 from course.

source dataset: https://www.tensorflow.org/tutorials/load_data/images


Target : Accuracy > 80%

Tips:
1.transforms.compose的Data Augmentation不一定要用
```python
transform_train = transforms.Compose(
    [
         transforms.ToPILImage(),
         transforms.Resize((height,width)), 
         transforms.ToTensor(),
    ]
)
```


2.height = 224，width =224

3.val_ratio = 0.1

4.batch_size = 64

5.densenet169


## License

MIT



