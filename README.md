# EKT/EERNN code

To run: `python run.py -w <directory> {config,train,test,stat,...}`

If this code helps with your studies, please kindly cite the following publication:

```
@article{liu2019ekt,
  title={EKT: Exercise-aware Knowledge Tracing for Student Performance Prediction},
  author={Liu, Qi and Huang, Zhenya and Yin, Yu and Chen, Enhong and Xiong, Hui and Su, Yu and Hu, Guoping},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2019},
  publisher={IEEE}
}
```

Also, visit https://base.ustc.edu.cn for more of our works.

## Configure

```
python run.py -w ws/test config EKTA -h  # check parameters available
python run.py -w ws/test config EKTA <arguments>
```

## Train

Specify dataset to train (no dataset publicly available, but demo dataset is on the way)

```
python run.py -w ws/test train -d full -N 1
```

## Test

Test predicting result on sequeence #10000:

```
python run.py -w ws/test test -d full_test -s 0.10000
```

## Evaluation

Results are under `ws/test/results`. To evaluate:

```
python run.py stat ws/test/results/school.0.10000
```
