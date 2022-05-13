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
python main_pefl.py
python run_edge_test_protocol.py i  # i = 0, 1, 2...模拟不同edge
```
