# Notes
cm = Conv2D(128, (4,4))
cm = MaxPool()(cm)
cm = Conv2D(64, (4,4))(cm)
cm = MaxPool()(cm)

# can be written as
def conmax(f,k):
    x = Conv2D(f, kernel_size=k)
    x = MaxPool()(x)
    return x

cm = conmax(128, (4,4))
cm = conmax(64, (4,4))(cm)

# Multiple inputs and outputs can't be done in Sequential model
im = Input(shape=(100,200))
ls = LSTM(10)(im)
dl = Dense(5)(ls)
ix = Input(name='input_aux')
cc = concatenate([dl, ix])
x = Dense(32)(cc)
mo = Dense(1)(x)
ao = Dense(2)(cc)

m = Model(inputs=[ix, dl], outputs=[mo, ao])


