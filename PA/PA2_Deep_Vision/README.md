## PA2 Deep Vision

```bash
conda activate ./hw2_env
```

### Visualization

- Visualize the structure of network
```bash
cd cifar-10
python network.py
cd ../experiments
tensorboard --logdir .
```

![network_structure](assets/network_structure.png)

- Visualize and check the input data

```bash
cd cifar-10
python dataset.py
```

| Origin                                                       | 1                                                            | 2                                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20250416175923914](assets/image-20250416175923914.png) | ![image-20250416175938200](assets/image-20250416175938200.png) | ![image-20250416175952072](assets/image-20250416175952072.png) |



- Train network and visualize the curves

```bash
cd cifar-10
python train.py
cd ../experiments
tensorboard --logdir .
```

| Loss                                                         | Acc@top1                                                     | Acc@top5                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="assets/image-20250416192222017.png" alt="image-20250416192222017" style="zoom:150%;" /> | ![image-20250416192325477](assets/image-20250416192325477.png) | ![image-20250416192408652](assets/image-20250416192408652.png) |
