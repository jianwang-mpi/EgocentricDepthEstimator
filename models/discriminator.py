import torch
import torch.nn as nn

class DiscriminatorSimpleConv(nn.Module):

    def __init__(self, input_features, input_feature_h=16, output_features=2):
        super(DiscriminatorSimpleConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_features, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Dropout2d(),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Dropout2d(),
            nn.LeakyReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Dropout2d(),
            nn.LeakyReLU()
        )
        linear_input_size = 4 * input_feature_h * input_feature_h
        self.linear = nn.Linear(linear_input_size, output_features)


    def forward(self, feature):
        x = self.conv1(feature)
        x = self.conv2(x)
        x = self.conv3(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.linear(x)
        return x

if __name__ == '__main__':
    dis = DiscriminatorSimpleConv(1, input_feature_h=128)
    sample1 = torch.zeros([4, 1, 128, 128])

    out = dis(sample1)
    out = torch.sum(out)
    out.backward()
    print(out.shape)
