안녕하세요~!


baseline_real = [{'in_channels':3, 'out_channels':32, 'stride':1, 'kernel_size':3, 'padding':1, 'conv':'real'}, # shpae (-1,32,32,32)
             # first layer
             {'in_channels':32, 'out_channels':64, 'stride':1, 'kernel_size':3, 'padding':1, 'conv':'real'}, # output shape (-1,64,32,32), Normal
             {'in_channels':64, 'out_channels':128, 'stride':2, 'kernel_size':3, 'padding':1, 'conv':'real'}, # output shape (-1,128,16,16), Reduction
             {'in_channels':128, 'out_channels':128, 'stride':1, 'kernel_size':3, 'padding':1, 'conv':'real'}, # output shape (-1,128,16,16), Normal
             {'in_channels':128, 'out_channels':256, 'stride':2, 'kernel_size':3, 'padding':1, 'conv':'real'}, # output shape (-1,256,8,8), Reduction
             {'in_channels':256, 'out_channels':256, 'stride':1, 'kernel_size':3, 'padding':1, 'conv':'real'}, # output shape (-1,256,8,8), Normal
             {'in_channels':256, 'out_channels':512, 'stride':2, 'kernel_size':3, 'padding':1, 'conv':'real'}, # output shape (-1,512,4,4), Reduction
             {'in_channels':512, 'out_channels':512, 'stride':1, 'kernel_size':3, 'padding':1, 'conv':'real'}, # output shape (-1,512,4,4), Normal
             {'in_channels':512, 'out_channels':512, 'stride':1, 'kernel_size':3, 'padding':1, 'conv':'real'}, # output shape (-1,512,4,4), Normal
             {'in_channels':512, 'out_channels':1024, 'stride':1, 'kernel_size':3, 'padding':1, 'conv':'real'}, # output shape (-1,1024,4,4), Normal
             {'in_channels':1024, 'out_channels':1024, 'stride':1, 'kernel_size':4, 'padding':0, 'conv':'pool'}, # output shape (-1,1024,1,1), avgpool
             {'in_channels':1024, 'out_channels':10, 'stride':1, 'kernel_size':1, 'padding':0, 'conv':'real'},] # output shape (-1,10,1,1)

baseline_real = [{'conv':'real','in_channels':3, 'out_channels':32, 'stride':1}, # shpae (-1,32,32,32)
             # first layer
             {'conv':'real','in_channels':32, 'out_channels':64, 'stride':1, 'kernel_size':3, 'padding':1}, # output shape (-1,64,32,32), Normal
             {'conv':'real','in_channels':64, 'out_channels':128, 'stride':2, 'kernel_size':3, 'padding':1}, # output shape (-1,128,16,16), Reduction
             {'conv':'real','in_channels':128, 'out_channels':128, 'stride':1, 'kernel_size':3, 'padding':1}, # output shape (-1,128,16,16), Normal
             {'conv':'real','in_channels':128, 'out_channels':256, 'stride':2, 'kernel_size':3, 'padding':1}, # output shape (-1,256,8,8), Reduction
             {'conv':'real','in_channels':256, 'out_channels':256, 'stride':1, 'kernel_size':3, 'padding':1}, # output shape (-1,256,8,8), Normal
             {'conv':'real','in_channels':256, 'out_channels':512, 'stride':2, 'kernel_size':3, 'padding':1}, # output shape (-1,512,4,4), Reduction
             {'conv':'real','in_channels':512, 'out_channels':512, 'stride':1, 'kernel_size':3, 'padding':1}, # output shape (-1,512,4,4), Normal
             {'conv':'real','in_channels':512, 'out_channels':512, 'stride':1, 'kernel_size':3, 'padding':1}, # output shape (-1,512,4,4), Normal
             {'conv':'real','in_channels':512, 'out_channels':1024, 'stride':1, 'kernel_size':3, 'padding':1}, # output shape (-1,1024,4,4), Normal
             {'conv':'pool','stride':1, 'kernel_size':3}, # output shape (-1,1024,1,1), avgpool
             {'conv':'real','in_channels':1024, 'out_channels':10, 'stride':1, 'kernel_size':1, 'padding':0}] # output shape (-1,10,1,1)
