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
 
 03. 