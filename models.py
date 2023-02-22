from NODE.NODE import *
from torch.autograd import Variable
from torch.nn import functional as F
import scipy.stats as st


class VortexBiconvGaussian(ODEF):
    def __init__(self):
        super(VortexBiconvGaussian, self).__init__()
        bias = True
        self.enc_conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias)
        self.enc_conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias)
        self.enc_conv5 = nn.Conv2d(128, 128, kernel_size=[5, 5], stride=2, padding=0, bias=bias)
        self.enc_conv6 = nn.Conv2d(128, 128, kernel_size=[5, 5], stride=2, padding=0, bias=bias)
        self.enc_conv7 = nn.Conv2d(128, 128, kernel_size=[2, 2], stride=2, padding=0, bias=bias)

        # self.enc_bn1 = nn.BatchNorm2d(32)
        # self.enc_bn2 = nn.BatchNorm2d(64)
        # self.enc_bn3 = nn.BatchNorm2d(128)
        # self.enc_bn4 = nn.BatchNorm2d(128)
        # self.enc_bn5 = nn.BatchNorm2d(128)
        # self.enc_bn6 = nn.BatchNorm2d(128)
        # self.enc_bn7 = nn.BatchNorm2d(128)

        self.lin1 = nn.Linear(128 * 4 * 2, 256, bias=bias)
        self.lin2 = nn.Linear(256, 256, bias=bias)
        self.lin3 = nn.Linear(256, 128, bias=bias)

        self.dec_conv1 = nn.ConvTranspose2d(128, 128, kernel_size=[5, 3], stride=3, padding=0, output_padding=0,
                                            bias=bias)
        self.dec_conv2 = nn.ConvTranspose2d(128, 128, kernel_size=[7, 3], stride=3, padding=0, bias=bias)
        self.dec_conv3 = nn.ConvTranspose2d(128, 128, kernel_size=[6, 6], stride=2, padding=0, output_padding=0,
                                            bias=bias)
        self.dec_conv4 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=0, output_padding=0, bias=bias)
        self.dec_conv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=0, output_padding=0, bias=bias)
        self.dec_conv6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=0, bias=bias)
        self.dec_conv7 = nn.ConvTranspose2d(32, 2, kernel_size=3, stride=1, padding=0, bias=bias)

        # self.dec_bn1 = nn.BatchNorm2d(128)
        # self.dec_bn2 = nn.BatchNorm2d(128)
        # self.dec_bn3 = nn.BatchNorm2d(128)
        # self.dec_bn4 = nn.BatchNorm2d(128)
        # self.dec_bn5 = nn.BatchNorm2d(64)
        # self.dec_bn6 = nn.BatchNorm2d(32)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # Create gaussian kernels
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ker_size = 5
        self.sigma = 0.1
        self.kernel = Variable(Tensor(self.gkern(self.ker_size, self.sigma))).view(1, 1, self.ker_size,
                                                                                   self.ker_size).to(device).double()

    def gkern(self, kernlen=11, nsig=0.05):  # large nsig gives more freedom(pixels as agents), small nsig is more fluid
        """Returns a 2D Gaussian kernel."""
        x = np.linspace(-nsig, nsig, kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        ker = kern2d / kern2d.sum()
        return ker

    def forward(self, x):
        bs, nc, imgx, imgy = x.shape
        # print("x shape: ", x.shape)
        # print("bs: ", bs)
        x = x.view(bs, nc, imgx, imgy)
        # print("x in", x.size())
        # print(x)
        x = self.relu(self.enc_conv1(x))
        x = self.relu(self.enc_conv2(x))
        x = self.relu(self.enc_conv3(x))
        x = self.relu(self.enc_conv4(x))
        x = self.relu(self.enc_conv5(x))
        x = self.relu(self.enc_conv6(x))
        x = self.relu(self.enc_conv7(x))
        # print("after conv", x.size())

        x = x.view(bs, -1)
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.lin3(x)

        x = x.view(bs, 128, 1, 1)
        x = self.relu(self.dec_conv1(x))
        x = self.relu(self.dec_conv2(x))
        x = self.relu(self.dec_conv3(x))
        x = self.relu(self.dec_conv4(x))
        x = self.relu(self.dec_conv5(x))
        x = self.relu(self.dec_conv6(x))
        x = self.dec_conv7(x)
        # print("final", x.size())
        x = x.view(bs, nc, imgx, imgy)
        # Apply smoothing
        x_smooth_x = F.conv2d(x[:, 0, :, :].squeeze().view(bs, 1, imgx, imgy), self.kernel,
                              padding=int((self.ker_size - 1) / 2), ).view(bs, 1, imgx, imgy)
        x_smooth_x = F.conv2d(x_smooth_x.squeeze().view(bs, 1, imgx, imgy), self.kernel,
                              padding=int((self.ker_size - 1) / 2), ).view(bs, 1, imgx, imgy)

        x_smooth_y = F.conv2d(x[:, 1, :, :].squeeze().view(bs, 1, imgx, imgy), self.kernel,
                              padding=int((self.ker_size - 1) / 2)).view(bs, 1, imgx, imgy)
        x_smooth_y = F.conv2d(x_smooth_y.squeeze().view(bs, 1, imgx, imgy), self.kernel,
                              padding=int((self.ker_size - 1) / 2), ).view(bs, 1, imgx, imgy)

        x_smooth = torch.cat([x_smooth_x, x_smooth_y], 1)

        return x_smooth


class VortexConvGaussian(ODEF):
    def __init__(self):
        super(VortexConvGaussian, self).__init__()
        bias = True
        self.enc_conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias)
        self.enc_conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias)
        self.enc_conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0, bias=bias)
        self.enc_conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0, bias=bias)
        self.enc_conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0, bias=bias)

        self.lin1 = nn.Linear(128 * 4 * 2, 128, bias=bias)
        self.lin2 = nn.Linear(128, 256, bias=bias)
        self.lin3 = nn.Linear(256, 2 * 50 * 30, bias=bias)

        self.relu = nn.Tanh()

        # Create gaussian kernels
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ker_size = 5
        self.sigma = 0.1
        self.kernel = Variable(Tensor(self.gkern(self.ker_size, self.sigma))).view(1, 1, self.ker_size,
                                                                                   self.ker_size).to(device).double()

    def gkern(self, kernlen=11, nsig=0.05):  # large nsig gives more freedom(pixels as agents), small nsig is more fluid
        """Returns a 2D Gaussian kernel."""
        x = np.linspace(-nsig, nsig, kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        ker = kern2d / kern2d.sum()
        return ker

    def forward(self, x):
        bs, nc, imgx, imgy = x.shape
        # print("bs: ", bs)
        x = x.view(bs, nc, imgx, imgy)
        #print("x in", x)
        x = self.relu(self.enc_conv1(x))
        x = self.relu(self.enc_conv2(x))
        x = self.relu(self.enc_conv3(x))
        x = self.relu(self.enc_conv4(x))
        x = self.relu(self.enc_conv5(x))
        x = self.relu(self.enc_conv6(x))
        x = self.relu(self.enc_conv7(x))
        # print("after conv", x.size())

        x = x.view(bs, -1)
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.lin3(x)

        x = x.view(bs, nc, imgx, imgy)
        # Apply smoothing
        x_smooth_x = F.conv2d(x[:, 0, :, :].squeeze().view(bs, 1, imgx, imgy), self.kernel,
                              padding=int((self.ker_size - 1) / 2), ).view(bs, 1, imgx, imgy)
        x_smooth_x = F.conv2d(x_smooth_x.squeeze().view(bs, 1, imgx, imgy), self.kernel,
                              padding=int((self.ker_size - 1) / 2), ).view(bs, 1, imgx, imgy)

        x_smooth_y = F.conv2d(x[:, 1, :, :].squeeze().view(bs, 1, imgx, imgy), self.kernel,
                              padding=int((self.ker_size - 1) / 2)).view(bs, 1, imgx, imgy)
        x_smooth_y = F.conv2d(x_smooth_y.squeeze().view(bs, 1, imgx, imgy), self.kernel,
                              padding=int((self.ker_size - 1) / 2), ).view(bs, 1, imgx, imgy)

        x_smooth = torch.cat([x_smooth_x, x_smooth_y], 1)

        return x_smooth


# class DGConvGaussian(ODEF):
#     def __init__(self):
#         super(DGConvGaussian, self).__init__()
#         bias = True
#         self.enc_conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=0, bias=bias)
#         self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=bias)
#         self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias)
#         self.enc_conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias)
#         self.enc_conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0, bias=bias)
#         self.enc_conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0, bias=bias)
#         self.enc_conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0, bias=bias)
#
#         self.lin1 = nn.Linear(1792, 128, bias=bias)
#         self.lin2 = nn.Linear(128, 256, bias=bias)
#         self.lin3 = nn.Linear(256, 2 * 34 * 67, bias=bias)
#
#         self.relu = nn.Tanh()
#
#         # Create gaussian kernels
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.ker_size = 5
#         self.sigma = 0.1
#         self.kernel = Variable(Tensor(self.gkern(self.ker_size, self.sigma))).view(1, 1, self.ker_size,
#                                                                                    self.ker_size).to(device).double()
#
#     def gkern(self, kernlen=11, nsig=0.05):  # large nsig gives more freedom(pixels as agents), small nsig is more fluid
#         """Returns a 2D Gaussian kernel."""
#         x = np.linspace(-nsig, nsig, kernlen + 1)
#         kern1d = np.diff(st.norm.cdf(x))
#         kern2d = np.outer(kern1d, kern1d)
#         ker = kern2d / kern2d.sum()
#         return ker
#
#     def forward(self, x):
#         bs, nc, imgx, imgy = x.shape
#         # print("bs: ", bs)
#         x = x.view(bs, nc, imgx, imgy)
#         # print("x in", x.size())
#         x = self.relu(self.enc_conv1(x))
#         x = self.relu(self.enc_conv2(x))
#         x = self.relu(self.enc_conv3(x))
#         x = self.relu(self.enc_conv4(x))
#         x = self.relu(self.enc_conv5(x))
#         x = self.relu(self.enc_conv6(x))
#         x = self.relu(self.enc_conv7(x))
#         # print("after conv", x.size())
#
#         x = x.view(bs, -1)
#         x = self.relu(self.lin1(x))
#         x = self.relu(self.lin2(x))
#         x = self.lin3(x)
#
#         x = x.view(bs, nc, imgx, imgy)
#         # Apply smoothing
#         x_smooth_x = F.conv2d(x[:, 0, :, :].squeeze().view(bs, 1, imgx, imgy), self.kernel,
#                               padding=int((self.ker_size - 1) / 2), ).view(bs, 1, imgx, imgy)
#         x_smooth_x = F.conv2d(x_smooth_x.squeeze().view(bs, 1, imgx, imgy), self.kernel,
#                               padding=int((self.ker_size - 1) / 2), ).view(bs, 1, imgx, imgy)
#
#         x_smooth_y = F.conv2d(x[:, 1, :, :].squeeze().view(bs, 1, imgx, imgy), self.kernel,
#                               padding=int((self.ker_size - 1) / 2)).view(bs, 1, imgx, imgy)
#         x_smooth_y = F.conv2d(x_smooth_y.squeeze().view(bs, 1, imgx, imgy), self.kernel,
#                               padding=int((self.ker_size - 1) / 2), ).view(bs, 1, imgx, imgy)
#
#         x_smooth = torch.cat([x_smooth_x, x_smooth_y], 1)
#
#         return x_smooth

class DGConvGaussian(ODEF):
    def __init__(self):
        super(DGConvGaussian, self).__init__()
        bias = True
        self.enc_conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=bias)
        self.enc_conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=bias)
        self.enc_conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=0, bias=bias)

        self.lin1 = nn.Linear(1600, 128, bias=bias)
        self.lin3 = nn.Linear(128, 2 * 50 * 50, bias=bias)

        self.relu = nn.Tanh()

        # Create gaussian kernels
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ker_size = 5
        self.sigma = 0.1
        self.kernel = Variable(Tensor(self.gkern(self.ker_size, self.sigma))).view(1, 1, self.ker_size,
                                                                                   self.ker_size).to(device).double()

    def gkern(self, kernlen=11, nsig=0.05):  # large nsig gives more freedom(pixels as agents), small nsig is more fluid
        """Returns a 2D Gaussian kernel."""
        x = np.linspace(-nsig, nsig, kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        ker = kern2d / kern2d.sum()
        return ker

    def forward(self, x):
        bs, nc, imgx, imgy = x.shape
        # print("bs: ", bs)
        x = x.view(bs, nc, imgx, imgy)
        # print("x in", x.size())
        x = self.relu(self.enc_conv1(x))
        x = self.relu(self.enc_conv2(x))
        x = self.relu(self.enc_conv3(x))
        x = self.relu(self.enc_conv4(x))
        x = self.relu(self.enc_conv5(x))

        x = x.view(bs, -1)
        x = self.relu(self.lin1(x))
        x = self.lin3(x)

        x = x.view(bs, nc, imgx, imgy)
        # Apply smoothing
        x_smooth_x = F.conv2d(x[:, 0, :, :].squeeze().view(bs, 1, imgx, imgy), self.kernel,
                              padding=int((self.ker_size - 1) / 2), ).view(bs, 1, imgx, imgy)

        x_smooth_y = F.conv2d(x[:, 1, :, :].squeeze().view(bs, 1, imgx, imgy), self.kernel,
                              padding=int((self.ker_size - 1) / 2)).view(bs, 1, imgx, imgy)

        x_smooth = torch.cat([x_smooth_x, x_smooth_y], 1)

        return x_smooth


class VortexConvGaussianSquare(ODEF):
    def __init__(self):
        super(VortexConvGaussianSquare, self).__init__()
        bias = True
        self.enc_conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=bias)
        self.enc_conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=bias)
        self.enc_conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0, bias=bias)

        self.lin1 = nn.Linear(1152, 128, bias=bias)
        self.lin3 = nn.Linear(128, 2 * 30 * 30, bias=bias)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # Create gaussian kernels
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ker_size = 5
        self.sigma = 0.1
        self.kernel = Variable(Tensor(self.gkern(self.ker_size, self.sigma))).view(1, 1, self.ker_size,
                                                                                   self.ker_size).to(device).double()

    def gkern(self, kernlen=11, nsig=0.05):  # large nsig gives more freedom(pixels as agents), small nsig is more fluid
        """Returns a 2D Gaussian kernel."""
        x = np.linspace(-nsig, nsig, kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        ker = kern2d / kern2d.sum()
        return ker

    def forward(self, x):
        bs, nc, imgx, imgy = x.shape
        # print("bs: ", bs)
        x = x.view(bs, nc, imgx, imgy)
        # print("x in", x.size())
        x = self.tanh(self.enc_conv1(x))
        x = self.tanh(self.enc_conv2(x))
        x = self.tanh(self.enc_conv3(x))
        x = self.tanh(self.enc_conv4(x))
        x = self.enc_conv5(x)

        x = x.view(bs, -1)
        x = self.relu(self.lin1(x))
        x = self.lin3(x)

        x = x.view(bs, nc, imgx, imgy)
        # Apply smoothing
        x_smooth_x = F.conv2d(x[:, 0, :, :].squeeze().view(bs, 1, imgx, imgy), self.kernel,
                              padding=int((self.ker_size - 1) / 2), ).view(bs, 1, imgx, imgy)

        x_smooth_y = F.conv2d(x[:, 1, :, :].squeeze().view(bs, 1, imgx, imgy), self.kernel,
                              padding=int((self.ker_size - 1) / 2)).view(bs, 1, imgx, imgy)

        x_smooth = torch.cat([x_smooth_x, x_smooth_y], 1)

        return x_smooth


class NOAAConvGaussian(ODEF):
    def __init__(self):
        super(NOAAConvGaussian, self).__init__()
        bias = True
        self.enc_conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=bias)
        self.enc_conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=bias)
        self.enc_conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=0, bias=bias)

        self.lin1 = nn.Linear(1600, 128, bias=bias)
        self.lin3 = nn.Linear(128, 2 * 50 * 50, bias=bias)

        self.relu = nn.Tanh()

        # Create gaussian kernels
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.ker_size = 5
        self.sigma = 0.1
        self.kernel = Variable(Tensor(self.gkern(self.ker_size, self.sigma))).view(1, 1, self.ker_size,
                                                                                   self.ker_size).to(device).double()

    def gkern(self, kernlen=11, nsig=0.05):  # large nsig gives more freedom(pixels as agents), small nsig is more fluid
        """Returns a 2D Gaussian kernel."""
        x = np.linspace(-nsig, nsig, kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        ker = kern2d / kern2d.sum()
        return ker

    def forward(self, x):
        bs, nc, imgx, imgy = x.shape
        # print("bs: ", bs)
        x = x.view(bs, nc, imgx, imgy)
        # print("x in", x.dtype)
        x = self.relu(self.enc_conv1(x))
        x = self.relu(self.enc_conv2(x))
        x = self.relu(self.enc_conv3(x))
        x = self.relu(self.enc_conv4(x))
        x = self.relu(self.enc_conv5(x))

        x = x.view(bs, -1)
        x = self.relu(self.lin1(x))
        x = self.lin3(x)

        x = x.view(bs, nc, imgx, imgy)
        # Apply smoothing
        x_smooth_x = F.conv2d(x[:, 0, :, :].squeeze().view(bs, 1, imgx, imgy), self.kernel,
                              padding=int((self.ker_size - 1) / 2), ).view(bs, 1, imgx, imgy)

        x_smooth_y = F.conv2d(x[:, 1, :, :].squeeze().view(bs, 1, imgx, imgy), self.kernel,
                              padding=int((self.ker_size - 1) / 2)).view(bs, 1, imgx, imgy)

        x_smooth = torch.cat([x_smooth_x, x_smooth_y], 1)

        return x_smooth


class NOAAConvGaussianNorm(ODEF):
    def __init__(self):
        super(NOAAConvGaussianNorm, self).__init__()
        bias = True
        self.enc_conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=bias)
        self.enc_conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=bias)
        self.enc_conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=0, bias=bias)

        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_bn2 = nn.BatchNorm2d(64)
        self.enc_bn3 = nn.BatchNorm2d(128)
        self.enc_bn4 = nn.BatchNorm2d(128)
        self.enc_bn5 = nn.BatchNorm2d(64)

        self.lin1 = nn.Linear(1600, 128, bias=bias)
        self.lin3 = nn.Linear(128, 2 * 50 * 50, bias=bias)

        self.relu = nn.Tanh()

        # Create gaussian kernels
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.ker_size = 5
        self.sigma = 0.1
        self.kernel = Variable(Tensor(self.gkern(self.ker_size, self.sigma))).view(1, 1, self.ker_size,
                                                                                   self.ker_size).to(device).double()

    def gkern(self, kernlen=11, nsig=0.05):  # large nsig gives more freedom(pixels as agents), small nsig is more fluid
        """Returns a 2D Gaussian kernel."""
        x = np.linspace(-nsig, nsig, kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        ker = kern2d / kern2d.sum()
        return ker

    def forward(self, x):
        bs, nc, imgx, imgy = x.shape
        # print("bs: ", bs)
        x = x.view(bs, nc, imgx, imgy)
        # print("x in", x.dtype)
        x = self.relu(self.enc_bn1(self.enc_conv1(x)))
        x = self.relu(self.enc_bn2(self.enc_conv2(x)))
        x = self.relu(self.enc_bn3(self.enc_conv3(x)))
        x = self.relu(self.enc_bn4(self.enc_conv4(x)))
        x = self.relu(self.enc_bn5(self.enc_conv5(x)))

        x = x.view(bs, -1)

        x = self.relu(self.lin1(x))
        x = self.lin3(x)

        x = x.view(bs, nc, imgx, imgy)
        # Apply smoothing
        x_smooth_x = F.conv2d(x[:, 0, :, :].squeeze().view(bs, 1, imgx, imgy), self.kernel,
                              padding=int((self.ker_size - 1) / 2), ).view(bs, 1, imgx, imgy)

        x_smooth_y = F.conv2d(x[:, 1, :, :].squeeze().view(bs, 1, imgx, imgy), self.kernel,
                              padding=int((self.ker_size - 1) / 2)).view(bs, 1, imgx, imgy)

        x_smooth = torch.cat([x_smooth_x, x_smooth_y], 1)

        return x_smooth


class ChaoticGaussianNorm(ODEF):
    # for size 80 by 80
    def __init__(self):
        super(ChaoticGaussianNorm, self).__init__()
        bias = True
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=bias)
        self.enc_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=bias)
        self.enc_conv5 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=0, bias=bias)

        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_bn3 = nn.BatchNorm2d(64)
        self.enc_bn4 = nn.BatchNorm2d(64)
        self.enc_bn5 = nn.BatchNorm2d(16)

        self.lin1 = nn.Linear(1296, 128, bias=bias)
        self.lin3 = nn.Linear(128, 80 * 80, bias=bias)

        self.relu = nn.Tanh()

        # Create gaussian kernels
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ker_size = 5
        self.sigma = 0.1
        self.kernel = Variable(Tensor(self.gkern(self.ker_size, self.sigma))).view(1, 1, self.ker_size,
                                                                                   self.ker_size).to(device).double()

    def gkern(self, kernlen=11, nsig=0.05):  # large nsig gives more freedom(pixels as agents), small nsig is more fluid
        """Returns a 2D Gaussian kernel."""
        x = np.linspace(-nsig, nsig, kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        ker = kern2d / kern2d.sum()
        return ker

    def forward(self, x):
        bs, nc, imgx, imgy = x.shape
        # print("bs: ", bs)
        x = x.view(bs, nc, imgx, imgy)
        # print("x in", x.dtype)
        x = self.relu(self.enc_bn1(self.enc_conv1(x)))
        x = self.relu(self.enc_bn2(self.enc_conv2(x)))
        x = self.relu(self.enc_bn3(self.enc_conv3(x)))
        x = self.relu(self.enc_bn4(self.enc_conv4(x)))
        x = self.relu(self.enc_bn5(self.enc_conv5(x)))

        x = x.view(bs, -1)

        x = self.relu(self.lin1(x))
        x = self.lin3(x)

        x = x.view(bs, nc, imgx, imgy)
        # Apply smoothing
        x_smooth_x = F.conv2d(x[:, 0, :, :].squeeze().view(bs, 1, imgx, imgy), self.kernel,
                              padding=int((self.ker_size - 1) / 2), ).view(bs, 1, imgx, imgy)

        return x_smooth_x


class GaussianNet(ODEF):
    def __init__(self):
        super(GaussianNet, self).__init__()
        bias = True
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=bias)
        self.enc_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=bias)
        self.enc_conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=0, bias=bias)

        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_bn3 = nn.BatchNorm2d(64)
        self.enc_bn4 = nn.BatchNorm2d(64)
        self.enc_bn5 = nn.BatchNorm2d(32)

        self.lin1 = nn.Linear(288, 128, bias=bias)
        self.lin3 = nn.Linear(128, 30 * 30, bias=bias)

        self.relu = nn.Tanh()

        # Create gaussian kernels
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ker_size = 5
        self.sigma = 0.1
        self.kernel = Variable(Tensor(self.gkern(self.ker_size, self.sigma))).view(1, 1, self.ker_size,
                                                                                   self.ker_size).to(device).double()

    def gkern(self, kernlen=11, nsig=0.05):  # large nsig gives more freedom(pixels as agents), small nsig is more fluid
        """Returns a 2D Gaussian kernel."""
        x = np.linspace(-nsig, nsig, kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        ker = kern2d / kern2d.sum()
        return ker

    def forward(self, x):
        bs, nc, imgx, imgy = x.shape
        # print("bs: ", bs)
        x = x.view(bs, nc, imgx, imgy)
        # print("x in", x.dtype)
        x = self.relu(self.enc_bn1(self.enc_conv1(x)))
        x = self.relu(self.enc_bn2(self.enc_conv2(x)))
        x = self.relu(self.enc_bn3(self.enc_conv3(x)))
        x = self.relu(self.enc_bn4(self.enc_conv4(x)))
        x = self.relu(self.enc_bn5(self.enc_conv5(x)))

        x = x.view(bs, -1)

        x = self.relu(self.lin1(x))
        x = self.lin3(x)

        x = x.view(bs, nc, imgx, imgy)
        # Apply smoothing
        x_smooth_x = F.conv2d(x[:, 0, :, :].squeeze().view(bs, 1, imgx, imgy), self.kernel,
                              padding=int((self.ker_size - 1) / 2), ).view(bs, 1, imgx, imgy)

        return x_smooth_x


class ChaoticGaussian40by40Norm(ODEF):
    # for size 40 by 40
    def __init__(self):
        super(ChaoticGaussian40by40Norm, self).__init__()
        bias = True
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.enc_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=bias)
        self.enc_conv5 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=0, bias=bias)

        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_bn3 = nn.BatchNorm2d(64)
        self.enc_bn4 = nn.BatchNorm2d(64)
        self.enc_bn5 = nn.BatchNorm2d(16)

        self.lin1 = nn.Linear(1024, 128, bias=bias)
        self.lin3 = nn.Linear(128, 40 * 40, bias=bias)

        self.relu = nn.Tanh()

        # Create gaussian kernels
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.ker_size = 5
        self.sigma = 0.1
        self.kernel = Variable(Tensor(self.gkern(self.ker_size, self.sigma))).view(1, 1, self.ker_size,
                                                                                   self.ker_size).to(device).double()

    def gkern(self, kernlen=11, nsig=0.05):  # large nsig gives more freedom(pixels as agents), small nsig is more fluid
        """Returns a 2D Gaussian kernel."""
        x = np.linspace(-nsig, nsig, kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        ker = kern2d / kern2d.sum()
        return ker

    def forward(self, x):
        bs, nc, imgx, imgy = x.shape
        # print("bs: ", bs)
        x = x.view(bs, nc, imgx, imgy)
        # print("x in", x.dtype)
        x = self.relu(self.enc_bn1(self.enc_conv1(x)))
        x = self.relu(self.enc_bn2(self.enc_conv2(x)))
        x = self.relu(self.enc_bn3(self.enc_conv3(x)))
        x = self.relu(self.enc_bn4(self.enc_conv4(x)))
        x = self.relu(self.enc_bn5(self.enc_conv5(x)))

        x = x.view(bs, -1)
        # print(x.shape)
        x = self.relu(self.lin1(x))
        x = self.lin3(x)

        x = x.view(bs, nc, imgx, imgy)
        # Apply smoothing
        x_smooth_x = F.conv2d(x[:, 0, :, :].squeeze().view(bs, 1, imgx, imgy), self.kernel,
                              padding=int((self.ker_size - 1) / 2), ).view(bs, 1, imgx, imgy)

        return x_smooth_x


class ChaoticGaussian40by40Norm2(ODEF):
    # for size 40 by 40
    def __init__(self):
        super(ChaoticGaussian40by40Norm2, self).__init__()
        bias = True
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.enc_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=bias)
        self.enc_conv5 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=0, bias=bias)

        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_bn3 = nn.BatchNorm2d(64)
        self.enc_bn4 = nn.BatchNorm2d(64)
        self.enc_bn5 = nn.BatchNorm2d(16)

        self.lin1 = nn.Linear(1024, 256, bias=bias)
        self.lin3 = nn.Linear(256, 40 * 40, bias=bias)

        self.relu = nn.Tanh()

        # Create gaussian kernels
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ker_size = 3
        self.sigma = 0.3
        self.kernel = Variable(Tensor(self.gkern(self.ker_size, self.sigma))).view(1, 1, self.ker_size,
                                                                                   self.ker_size).to(device).double()

    def gkern(self, kernlen=11, nsig=0.05):  # large nsig gives more freedom(pixels as agents), small nsig is more fluid
        """Returns a 2D Gaussian kernel."""
        x = np.linspace(-nsig, nsig, kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        ker = kern2d / kern2d.sum()
        return ker

    def forward(self, x):
        bs, nc, imgx, imgy = x.shape
        # print("bs: ", bs)
        x = x.view(bs, nc, imgx, imgy)
        # print("x in", x.dtype)
        x = self.relu(self.enc_bn1(self.enc_conv1(x)))
        x = self.relu(self.enc_bn2(self.enc_conv2(x)))
        x = self.relu(self.enc_bn3(self.enc_conv3(x)))
        x = self.relu(self.enc_bn4(self.enc_conv4(x)))
        x = self.relu(self.enc_bn5(self.enc_conv5(x)))

        x = x.view(bs, -1)
        # print(x.shape)
        x = self.relu(self.lin1(x))
        x = self.lin3(x)

        x = x.view(bs, nc, imgx, imgy)
        # Apply smoothing
        x_smooth_x = F.conv2d(x[:, 0, :, :].squeeze().view(bs, 1, imgx, imgy), self.kernel,
                              padding=int((self.ker_size - 1) / 2), ).view(bs, 1, imgx, imgy)

        return x_smooth_x






class ChaoticGaussian40by40Norm(ODEF):
    # for size 40 by 40
    def __init__(self):
        super(ChaoticGaussian40by40Norm, self).__init__()
        bias = True
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.enc_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=bias)
        self.enc_conv5 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=0, bias=bias)

        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_bn3 = nn.BatchNorm2d(64)
        self.enc_bn4 = nn.BatchNorm2d(64)
        self.enc_bn5 = nn.BatchNorm2d(16)

        self.lin1 = nn.Linear(1024, 128, bias=bias)
        self.lin3 = nn.Linear(128, 40 * 40, bias=bias)

        self.relu = nn.Tanh()

        # Create gaussian kernels
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.ker_size = 5
        self.sigma = 0.1
        self.kernel = Variable(Tensor(self.gkern(self.ker_size, self.sigma))).view(1, 1, self.ker_size,
                                                                                   self.ker_size).to(device).double()

    def gkern(self, kernlen=11, nsig=0.05):  # large nsig gives more freedom(pixels as agents), small nsig is more fluid
        """Returns a 2D Gaussian kernel."""
        x = np.linspace(-nsig, nsig, kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        ker = kern2d / kern2d.sum()
        return ker

    def forward(self, x):
        bs, nc, imgx, imgy = x.shape
        # print("bs: ", bs)
        x = x.view(bs, nc, imgx, imgy)
        # print("x in", x.dtype)
        x = self.relu(self.enc_bn1(self.enc_conv1(x)))
        x = self.relu(self.enc_bn2(self.enc_conv2(x)))
        x = self.relu(self.enc_bn3(self.enc_conv3(x)))
        x = self.relu(self.enc_bn4(self.enc_conv4(x)))
        x = self.relu(self.enc_bn5(self.enc_conv5(x)))
        
        x = x.view(bs, -1)
        # print(x.shape)
        x = self.relu(self.lin1(x))
        x = self.lin3(x)

        x = x.view(bs, nc, imgx, imgy)
        # Apply smoothing
        x_smooth_x = F.conv2d(x[:, 0, :, :].squeeze().view(bs, 1, imgx, imgy), self.kernel,
                              padding=int((self.ker_size - 1) / 2), ).view(bs, 1, imgx, imgy)

        return x_smooth_x


class ChaoticGaussian40by40Norm2(ODEF):
    # for size 40 by 40
    def __init__(self):
        super(ChaoticGaussian40by40Norm2, self).__init__()
        bias = True
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.enc_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=bias)
        self.enc_conv5 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=0, bias=bias)

        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_bn3 = nn.BatchNorm2d(64)
        self.enc_bn4 = nn.BatchNorm2d(64)
        self.enc_bn5 = nn.BatchNorm2d(16)

        self.lin1 = nn.Linear(1024, 256, bias=bias)
        self.lin3 = nn.Linear(256, 40 * 40, bias=bias)

        self.relu = nn.Tanh()

        # Create gaussian kernels
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ker_size = 3
        self.sigma = 0.3
        self.kernel = Variable(Tensor(self.gkern(self.ker_size, self.sigma))).view(1, 1, self.ker_size,
                                                                                   self.ker_size).to(device).double()

    def gkern(self, kernlen=11, nsig=0.05):  # large nsig gives more freedom(pixels as agents), small nsig is more fluid
        """Returns a 2D Gaussian kernel."""
        x = np.linspace(-nsig, nsig, kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        ker = kern2d / kern2d.sum()
        return ker

    def forward(self, x):
        bs, nc, imgx, imgy = x.shape
        # print("bs: ", bs)
        x = x.view(bs, nc, imgx, imgy)
        # print("x in", x.dtype)
        x = self.relu(self.enc_bn1(self.enc_conv1(x)))
        x = self.relu(self.enc_bn2(self.enc_conv2(x)))
        x = self.relu(self.enc_bn3(self.enc_conv3(x)))
        x = self.relu(self.enc_bn4(self.enc_conv4(x)))
        x = self.relu(self.enc_bn5(self.enc_conv5(x)))
        
        x = x.view(bs, -1)
        # print(x.shape)
        x = self.relu(self.lin1(x))
        x = self.lin3(x)

        x = x.view(bs, nc, imgx, imgy)
        # Apply smoothing
        x_smooth_x = F.conv2d(x[:, 0, :, :].squeeze().view(bs, 1, imgx, imgy), self.kernel,
                              padding=int((self.ker_size - 1) / 2), ).view(bs, 1, imgx, imgy)

        return x_smooth_x


class ChaoticFlowField40by40Norm(ODEF):
    # for size 40 by 40
    def __init__(self):
        super(ChaoticFlowField40by40Norm, self).__init__()
        bias = True
        self.enc_conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.enc_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=bias)
        self.enc_conv5 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=0, bias=bias)

        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_bn3 = nn.BatchNorm2d(64)
        self.enc_bn4 = nn.BatchNorm2d(64)
        self.enc_bn5 = nn.BatchNorm2d(16)

        self.lin1 = nn.Linear(1024, 256, bias=bias)
        self.lin3 = nn.Linear(256, 2 * 40 * 40, bias=bias)

        self.relu = nn.Tanh()

        # Create gaussian kernels
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ker_size = 3
        self.sigma = 0.3
        self.kernel = Variable(Tensor(self.gkern(self.ker_size, self.sigma))).view(1, 1, self.ker_size,
                                                                                   self.ker_size).to(device).double()

    def gkern(self, kernlen=11, nsig=0.05):  # large nsig gives more freedom(pixels as agents), small nsig is more fluid
        """Returns a 2D Gaussian kernel."""
        x = np.linspace(-nsig, nsig, kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        ker = kern2d / kern2d.sum()
        return ker

    def forward(self, x):
        bs, nc, imgx, imgy = x.shape
        # print("bs: ", bs)
        x = x.view(bs, nc, imgx, imgy)
        # print("x in", x.dtype)
        x = self.relu(self.enc_bn1(self.enc_conv1(x)))
        x = self.relu(self.enc_bn2(self.enc_conv2(x)))
        x = self.relu(self.enc_bn3(self.enc_conv3(x)))
        x = self.relu(self.enc_bn4(self.enc_conv4(x)))
        x = self.relu(self.enc_bn5(self.enc_conv5(x)))

        x = x.view(bs, -1)
        # print(x.shape)
        x = self.relu(self.lin1(x))
        x = self.lin3(x)

        x = x.view(bs, nc, imgx, imgy)

        # Apply smoothing
        x_smooth_x = F.conv2d(x[:, 0, :, :].squeeze().view(bs, 1, imgx, imgy), self.kernel,
                              padding=int((self.ker_size - 1) / 2), ).view(bs, 1, imgx, imgy)

        x_smooth_y = F.conv2d(x[:, 1, :, :].squeeze().view(bs, 1, imgx, imgy), self.kernel,
                              padding=int((self.ker_size - 1) / 2)).view(bs, 1, imgx, imgy)

        x_smooth = torch.cat([x_smooth_x, x_smooth_y], 1)
        return x_smooth

class ChaoticVorticity40by40Norm_noGaussian(ODEF):
    # for size 40 by 40
    def __init__(self):
        super(ChaoticVorticity40by40Norm_noGaussian, self).__init__()
        bias = True
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.enc_conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias)
        self.enc_conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0, bias=bias)
        self.enc_conv6 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=0, bias=bias)
        self.enc_conv7 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=0, bias=bias)

        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_bn3 = nn.BatchNorm2d(64)
        self.enc_bn4 = nn.BatchNorm2d(128)
        self.enc_bn5 = nn.BatchNorm2d(128)
        self.enc_bn6 = nn.BatchNorm2d(64)
        self.enc_bn7 = nn.BatchNorm2d(16)

        self.lin1 = nn.Linear(784, 512, bias=bias)
        self.lin3 = nn.Linear(512, 40 * 40, bias=bias)

        self.relu = nn.Tanh()

        # Create gaussian kernels
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        bs, nc, imgx, imgy = x.shape
        # print("bs: ", bs)
        x = x.view(bs, nc, imgx, imgy)
        # print("x in", x.dtype)
        x = self.relu(self.enc_bn1(self.enc_conv1(x)))
        x = self.relu(self.enc_bn2(self.enc_conv2(x)))
        x = self.relu(self.enc_bn3(self.enc_conv3(x)))
        x = self.relu(self.enc_bn4(self.enc_conv4(x)))
        x = self.relu(self.enc_bn5(self.enc_conv5(x)))
        x = self.relu(self.enc_bn6(self.enc_conv6(x)))
        x = self.relu(self.enc_bn7(self.enc_conv7(x)))

        x = x.view(bs, -1)
        x = self.relu(self.lin1(x))
        x = self.lin3(x)

        x = x.view(bs, nc, imgx, imgy)

        return x

