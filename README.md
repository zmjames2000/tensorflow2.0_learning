# tensorflow2.0_learning
learning tensorflow2.0 

01. fashion_mnist
    1. tensorflow 2.0 的框架
    2. 如何拆分数据 
        from sklearn.model_selection import train_test_split
    3. 如何归一化数据
        from sklearn.preprocessing import StandardScaler
        
        
02. callbacks
    callbacks = [
        keras.callbacks.TensorBoard(logdir),
        keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True),
        keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3),
    ]
    
    tensorboard --logdir=callbacks
 
 03. fetch_california_housing
 
 04. 激活函数
    1. sigmoid
    2. Leaky ReLU
    3. tanh
    4. Maxout
    5. ReLU
    6. ELU
    7. selu 是自带归一化的函数的
  
  05. 深度神经网络 方法1
    model = keras.models.Sequential()
    for _ in range(20):
        model.add(keras.layers.Dense(30, activation='relu', input_shape=x_train.shape[1:]))
    model.add(keras.layers.Dense(1)) 
    
   06.实现批归一化  BatchNormalization()
   07. 实现Dropout 
        model.add(keras.layers.AlphaDropout(rate=0.5))
   
   08. Wide and Deep 模型
    