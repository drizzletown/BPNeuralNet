
# BPNeuralNet
- BP神经网络的C++实现（Visual Studio 2017），支持多种激活函数，支持随机、批量、小批量训练

#使用方法：

##1. 定义神经网络各层的神经元个数
  `vector<int> layer_neuron_num = {2,10,1};
  
##2. 准备训练数据和实际目标值

```
  const int num = 11;
  double b[num][3] = { {2,5,10}, {3,6,18},{12,2,24},{1,6,6},{9,2,18},
			{8,12,96}, {4,7,28},{7,9,63},{1,10,10},{15,8,120},
			{220,3,660} };

  double tg[num] = {};
  for (int i = 0; i < num; i++) { tg[i] = b[i][2]; }

  MappingToN1_1 mping;
  double amin=0, amax=0;
  //将目标值映射到[-1,1]范围
  vector<Vector_xd> target = mping.map(tg, num, amin, amax);	
  for (int i = 0; i < num; i++) {
    Vector_xd x1(2); x1 << b[i][0],b[i][1]; x.push_back(x1);
  }
```

##3. 训练神经网络

```
  nn.setEpoch(3000);
  nn.setLearningRate(0.05);
  nn.setActFunctionType(ActFunctionType::TANH);
  nn.setTrainingType(TrainingType::MiniBatched);
  nn.training(x, target);
```

##4. 预测

```
  vector<Vector_xd> prd;
  nn.pred(x, prd);
  //将目标值从[-1,1]范围反向映射到实际范围
  mping.deMap(prd, amin, amax);
  //输出预测结果
  for (int i = 0; i < x.size(); i++) {
    cout << "输入数据：" << endl;
    for (int j = 0; j < x[i].size(); j++)
      cout << x[i](j) << " ";
    cout << endl;
    cout << "目标值-预测值：" << tg[i] << "\t" << prd[i](0) << endl;
  }
```
  
 
