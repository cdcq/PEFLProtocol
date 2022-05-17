# 测试方法
建议使用konsole等支持tab的终端模拟器，对于shell建议默认开启虚拟环境:

`fish`：在`~/.config/fish/config.fish`中添加`conda activate <虚拟环境>`


## 找protocol的bug
切换到`tests`目录下，然后依次在不同shell中运行：
```bash
python run_kgc.py
python run_cp.py
python run_sp.py
python run_edge_test_protocol.py i  # i = 0, 1, 2...模拟不同edge
```

## 正常运行, 或找ML部分bug
切换到`tests`目录下，然后依次在不同shell中运行：
```bash
python test_backdoor.py
```
## test_backdoor
```bash
python test_backdoor.py --params ./ML_utils/utils/mnist_params.yaml 

python test_backdoor.py --params ./ML_utils/utils/cifar_params.yaml #cifar采用预训练模型
```
攻击成功率即：样本里全是有毒的。有毒的预测正确的/总样本数 由Mytestpoison()
主任务上的acc：由Mytest()或者test_model()

投毒的client和round设置在test_backdoor.py里：
```python
 for round_id in range(MAX_ROUND):
    for edge_id in range(TRAINERS_COUNT):#0-9
        if edge_id in [1,3,2,8,9,0]: #投毒的4个[4 5 7 8]
            grads_list, local_loss = local_update(model=model, dataloader=edge_dataloaders[edge_id])
        else:
            if (round_id==2 and edge_id==4) or  (round_id%3 and edge_id==5) or (round_id==0 and edge_id==7) or (round_id==1 and edge_id==6):
                grads_list, local_loss = poison_local_update(edge_id=edge_id-4,model=model,target_model=model)
```