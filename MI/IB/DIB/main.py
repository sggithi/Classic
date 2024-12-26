###############################################
# Reference: https://github.com/yuxi120407/DIB
# CIFAR-10
##############################################
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)


trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=6)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=6)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


from utils import calculate_MI

def train(epoch):
    # Train시에 Loss에다가 Information Bottleneck 추가하는 정도
    train_loss = 0
    IXZ_loss = 0 # I(X;Z)에 대한 Loss
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # inputs: CIFAR-10 Image, targets: Class
        Z, outputs = model(inputs) # model's output & hidden layer Z
        loss = criterion(outputs, targets)

        with torch.no_grad():
            Z_numpy = Z.cpu().detach().numpy()
            k = squareform(pdist(Z_numpy, 'eculidean')) # Z의 Sample N개라면 그 중 두개씩 유클리디안 거리 => K (N, N)
            sigma_z = np.meean(np.mean(np.sort(k[:, :10], 1))) # 각 샘플 중에 가장 가까운 10개 선택해서 분산 계산산
            
            inputs_numpy = inputs.cpu().detach().numpy()
            inputs_numpy = inputs_numpy.reshape(inputs.shape[0],-1)
            k_input = squareform(pdist(inputs_numpy, 'euclidean'))
            sigma_input = np.mean(np.mean(np.sort(k_input[:, :10], 1))) # Input image에 대해서도 분산 계산
            
        # 계산된 sigma 기반으로 MI 계산하고, Loss에 비중 줘서 추가해주기
        IXZ = calculate_MI(inputs,Z,s_x=sigma_input,s_y=sigma_z)
        total_loss = loss + 0.01*IXZ
        total_loss.backward()
        optimizer.step()

        # ...

        