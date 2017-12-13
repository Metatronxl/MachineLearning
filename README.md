# MachineLearning
---
> 机器学习的过程记录

> 参考教材：Python Machine Learning Cookbook
---
### demo1：估算房屋价格
> 数据来源：sklearn.datasets  

> 主要算法：决策树 && AdaBoost算法
-  Conclusion:研究结果表明AdaBoost算法可以让误差更小,且解释方差分更接近1
-  adaBoost 的思想是将不同版本的算法结果进行组合,加权汇总获取最终结果
-  详解AdaBoost:http://www.jianshu.com/p/389d28f853c0
-  Scikit learn adaBoost参数详解 http://www.cnblogs.com/pinard/p/6136914.html

---
### demo2：评估共享单车的需求分布
> 数据来源：bike_day.csv

> 主要算法：随机森林

```
rf_regressor = RandomForestRegressor(n_estimators=1000,max_depth=10,min_samples_split=2)

explain:
      n_estimators是指评估器(estimator)的数量,表示随机森林需要使用的决策树数量
      max_depth是指每个决策树的最大深度
      min_samples_split是指决策树分裂一个节点需要用到的最小数据样本量
```
```
鉴于决策树容易过拟合的缺点，随机森林采用多个决策树的投票机制来改善决策树，我们假设随机森林使用了m棵决策树，那么就需要产生m个一定数量的样本集来训练每一棵树，如果用全样本去训练m棵决策树显然是不可取的，全样本训练忽视了局部样本的规律，对于模型的泛化能力是有害的
产生n个样本的方法采用Bootstraping法，这是一种有放回的抽样方法，产生n个样本
而最终结果采用Bagging的策略来获得，即多数投票机制

随机森林的生成方法：
1.从样本集中通过重采样的方式产生n个样本
2.假设样本特征数目为a，对n个样本选择a中的k个特征，用建立决策树的方式获得最佳分割点
3.重复m次，产生m棵决策树
4.多数投票机制来进行预测
（需要注意的一点是，这里m是指循环的次数，n是指样本的数目，n个样本构成训练的样本集，而m次循环中又会产生m个这样的样本集）
```



