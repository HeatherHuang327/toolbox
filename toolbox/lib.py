def build_num_branch():
    from tensorflow.keras import Sequential, Model, layers
    input_num = layers.Input(shape=(100,), name='input_num')

    o_num = layers.Dense(64, activation="relu", name='num_dense_1')(input_num)
    o_num = layers.Dense(32, activation="relu", name='num_dense_2')(o_num)
    o_num = layers.Dense(10, activation="relu", name='num_dense_3')(o_num)
    branch = Model(inputs=input_num, outputs=o_num, name='num_section')
    return branch
