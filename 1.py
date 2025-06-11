# utils.py 中的天气数据处理函数

# 将时间序列数据转换为模型可用的样本序列
def seq2instance(data, P, Q):
    num_step, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape=(num_sample, P, dims))
    y = np.zeros(shape=(num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i: i + P]
        y[i] = data[i + P: i + P + Q]
    return x, y

# 天气数据预处理专用函数
def weatherProcess(Weather, wtrain_steps, wval_steps, wtest_steps):
    # 按照训练集、验证集和测试集的比例划分天气数据
    wtrain = Weather[: wtrain_steps]
    wval = Weather[wtrain_steps: wtrain_steps + wval_steps]
    wtest = Weather[-wtest_steps:]

    # 将天气数据转换为序列样本
    trainX, trainY = seq2instance(wtrain, args.P, args.Q)
    wtrain = np.concatenate((trainX, trainY), axis=1).astype(np.int32)
    
    valX, valY = seq2instance(wval, args.P, args.Q)
    wval = np.concatenate((valX, valY), axis=1).astype(np.int32)
    
    testX, testY = seq2instance(wtest, args.P, args.Q)
    wtest = np.concatenate((testX, testY), axis=1).astype(np.int32)

    # 添加一个维度用于后续拼接多种天气指标
    wtrain = np.expand_dims(wtrain, axis=3)
    wval = np.expand_dims(wval, axis=3)
    wtest = np.expand_dims(wtest, axis=3)

    return wtrain, wval, wtest

# 主要的数据加载和处理函数
def loadData_weather(args):
    # 加载交通数据
    df = pd.read_hdf(args.traffic_file)
    Traffic = df.values
    
    # 划分交通数据的训练/验证/测试集
    num_step = df.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    train = Traffic[: train_steps]
    val = Traffic[train_steps: train_steps + val_steps]
    test = Traffic[-test_steps:]
    
    # 生成交通数据的输入序列和标签
    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX, valY = seq2instance(val, args.P, args.Q)
    testX, testY = seq2instance(test, args.P, args.Q)
    
    # 标准化交通数据
    mean, std = np.mean(trainX), np.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

    # 加载空间嵌入数据
    f = open(args.SE_file, mode='r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0]), int(temp[1])
    SE = np.zeros(shape=(N, dims), dtype=np.float32)
    for line in lines[1:]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index] = temp[1:]

    # 生成时间嵌入
    Time = df.index
    dayofweek = np.reshape(Time.weekday, newshape=(-1, 1))
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) // 300
    timeofday = np.reshape(timeofday, newshape=(-1, 1))
    Time = np.concatenate((dayofweek, timeofday), axis=-1)
    
    # 划分时间嵌入数据
    train = Time[: train_steps]
    val = Time[train_steps: train_steps + val_steps]
    test = Time[-test_steps:]
    
    # 生成时间嵌入序列
    trainTE = seq2instance(train, args.P, args.Q)
    trainTE = np.concatenate(trainTE, axis=1).astype(np.int32)
    valTE = seq2instance(val, args.P, args.Q)
    valTE = np.concatenate(valTE, axis=1).astype(np.int32)
    testTE = seq2instance(test, args.P, args.Q)
    testTE = np.concatenate(testTE, axis=1).astype(np.int32)

    # 以下是天气数据处理部分
    # 1. 加载各种天气数据
    dfrain = pd.read_hdf(args.weather_wind_speed_file)  # 风速数据
    WeatherRain = dfrain.values

    dfclouds = pd.read_hdf(args.weather_clouds_file)  # 云量/能见度数据
    Weatherclouds = dfclouds.values

    dfpressure = pd.read_hdf(args.weather_pressure_file)  # 气压数据
    Weatherpressure = dfpressure.values

    dftemp = pd.read_hdf(args.weather_visibility_file)  # 温度/能见度数据
    Weathertemp = dftemp.values

    # 2. 对气压数据进行标准化
    Pressuremean, Pressurestd = np.mean(Weatherpressure), np.std(Weatherpressure)
    Weatherpressure = (Weatherpressure - Pressuremean) / Pressurestd

    # 3. 划分天气数据的训练/验证/测试集
    wnum_step = dfrain.shape[0]
    wtrain_steps = round(args.train_ratio * wnum_step)
    wtest_steps = round(args.test_ratio * wnum_step)
    wval_steps = wnum_step - wtrain_steps - wtest_steps

    # 4. 处理各种天气数据
    trainRain, valRain, testRain = weatherProcess(WeatherRain, wtrain_steps, wval_steps, wtest_steps)
    trainclouds, valclouds, testclouds = weatherProcess(Weatherclouds, wtrain_steps, wval_steps, wtest_steps)
    trainpressure, valpressure, testpressure = weatherProcess(Weatherpressure, wtrain_steps, wval_steps, wtest_steps)
    traintemp, valtemp, testtemp = weatherProcess(Weathertemp, wtrain_steps, wval_steps, wtest_steps)

    # 5. 将不同天气指标合并为一个多通道的天气特征
    # 先合并风速和云量
    trainW = np.concatenate((trainRain, trainclouds), axis=3).astype(np.int32)
    valW = np.concatenate((valRain, valclouds), axis=3).astype(np.int32)
    testW = np.concatenate((testRain, testclouds), axis=3).astype(np.int32)
    # 再合并气压
    trainW = np.concatenate((trainW, trainpressure), axis=3).astype(np.int32)
    valW = np.concatenate((valW, valpressure), axis=3).astype(np.int32)
    testW = np.concatenate((testW, testpressure), axis=3).astype(np.int32)
    # 最后合并温度
    trainW = np.concatenate((trainW, traintemp), axis=3).astype(np.int32)
    valW = np.concatenate((valW, valtemp), axis=3).astype(np.int32)
    testW = np.concatenate((testW, testtemp), axis=3).astype(np.int32)

    # 返回所有处理好的数据
    return (trainX, trainTE, trainY, trainW, valX, valTE, valY, valW, testX, testTE, testY, testW,
            SE, mean, std)