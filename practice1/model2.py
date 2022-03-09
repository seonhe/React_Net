

class reactnet(nn.Module):

    def __init__(self, alpha=0.2, num_classes=1000):
        super(reactnet, self).__init__()
        
        self.feature = nn.ModuleList()
        for i in range(len(stage_out_channel)):
            if i == 0:

                expected_var = 1.0
                beta1 = 1. / expected_var ** 0.5
                expected_var += alpha ** 2
                beta2 = 1. / expected_var ** 0.5

                self.feature.append(firstconv3x3(3, stage_out_channel[i], 2))
            elif stage_out_channel[i-1] != stage_out_channel[i] and stage_out_channel[i] != 64:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], alpha, beta1, beta2, 2))
                # Reset expected var at a transition block
                expected_var = 1.0
                beta1 = 1. / expected_var ** 0.5
                expected_var += alpha ** 2
                beta2 = 1. / expected_var ** 0.5

            else:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], alpha, beta1, beta2, 1))
                
                expected_var += alpha ** 2
                beta1 = 1. / expected_var ** 0.5
                expected_var += alpha ** 2
                beta2 = 1. / expected_var ** 0.5

        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        for i, block in enumerate(self.feature):
            x = block(x)

        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x