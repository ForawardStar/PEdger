import torch

if __name__ == '__main__':
    # state_dict = torch.load('./checkpoints/IANet_CA_51.pth')
    # state_dict_new = {}
    # for key in state_dict.keys():
    #     state_dict_new[key.replace('up2', 'up1').replace('up3', 'up2').replace('up4', 'up3')] = state_dict[key]
    # torch.save(state_dict_new, './checkpoints/IANet_CA_51.pth')
    #
    # state_dict = torch.load('./checkpoints/NSNet_422.pth')
    # state_dict_new = {}
    # for key in state_dict.keys():
    #     state_dict_new[key.replace('up2', 'up1').replace('up3', 'up2').replace('up4', 'up3')] = state_dict[key]
    # torch.save(state_dict_new, './checkpoints/NSNet_422_new.pth')

    state_dict = torch.load('./checkpoints/FuseNet_MEF_404.pth')
    state_dict_new = {}
    for key in state_dict.keys():
        state_dict_new[key.replace('up3', 'up1').replace('up4', 'up2')] = state_dict[key]
    torch.save(state_dict_new, 'checkpoints/FuseNet_MEF_404.pth')

    # state_dict = torch.load('./checkpoints/FuseNet_FD_297.pth')
    # state_dict_new = {}
    # for key in state_dict.keys():
    #     state_dict_new[key.replace('up3', 'up1').replace('up4', 'up2')] = state_dict[key]
    # torch.save(state_dict_new, 'checkpoints/FuseNet_FD_297.pth')
